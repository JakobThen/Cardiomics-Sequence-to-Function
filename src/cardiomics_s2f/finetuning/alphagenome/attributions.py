"""
AlphaGenome Attributions
========================

Nucleotide-level attributions for fine-tuned AlphaGenome heads.

* `compute_deepshap_attributions`: DeepSHAP-style attributions (gradient of the
  input sequence minus the mean gradient of shuffled reference sequences) for
  one track, or the mean over a head's tracks, at 1 bp resolution.
* `one_hot_encode_sequences`: fast one-hot encoding of DNA strings.
* `normalize_attributions_by_local_activity`: divides attributions by their
  Gaussian-smoothed local activity (the normalisation used for Borzoi
  attributions), so motifs in low- and high-activity regions become comparable.

Requires JAX, PyTorch and ``tangermeme`` (dinucleotide shuffles).
"""

from __future__ import annotations

from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from tangermeme.ersatz import dinucleotide_shuffle


def _get_1bp_key(output_dict: dict, return_scaled: bool) -> str:
    """Select the correct 1bp resolution key from a head output dict.

    Args:
        output_dict:   Head output dictionary with resolution keys.
        return_scaled: If True, prefer scaled predictions.

    Returns:
        The matching key string.

    Raises:
        KeyError: If no 1bp key is found.
    """
    preferred = "scaled_predictions_1bp" if return_scaled else "predictions_1bp"
    fallback  = "predictions_1bp"        if return_scaled else "scaled_predictions_1bp"

    if preferred in output_dict:
        return preferred
    if fallback in output_dict:
        return fallback

    raise KeyError(
        f"No 1bp prediction key found in head output. "
        f"Tried: '{preferred}', '{fallback}'. "
        f"Available keys: {list(output_dict.keys())}"
    )


def _resolve_track_index(
    output_index: int | None,
    output_key:   str | None,
    track_names:  list[str] | None,
    n_tracks:     int,
) -> int | None:
    """Resolve the target track to an integer index.

    Priority: output_key (name-based) > output_index (int-based) > None (mean).

    Args:
        output_index: Integer track index (0-based).
        output_key:   Track name string. Requires track_names.
        track_names:  Ordered list of track names matching the last axis of the
                      prediction array.
        n_tracks:     Total number of tracks (for bounds checking).

    Returns:
        Resolved integer index, or None meaning mean over all tracks.
    """
    if output_key is not None:
        if track_names is None:
            raise ValueError(
                "output_key was provided but track_names is None. "
                "Pass track_names=list(tracks_metadata[head_name]['name']) "
                "so the name can be resolved to an index."
            )
        if output_key not in track_names:
            raise ValueError(
                f"Track '{output_key}' not found in track_names. "
                f"Available tracks: {track_names}"
            )
        return track_names.index(output_key)

    if output_index is not None:
        if output_index >= n_tracks:
            raise ValueError(
                f"output_index={output_index} out of range. "
                f"Head has {n_tracks} tracks (0-indexed)."
            )
        return output_index

    # Neither specified — mean over all tracks
    return None


def compute_deepshap_attributions(
    model,
    sequence,
    organism_index,
    *,
    head_name:      str | None       = None,
    output_index:   int | None       = None,
    output_key:     str | None       = None,
    track_names:    list[str] | None = None,
    return_scaled:  bool             = True,
    n_references:   int              = 20,
    reference_type: str              = 'dinucleotide_shuffle',
    random_state:   int | None       = None,
    center_mask_size:    int | None       = None
):
    """Computes DeepSHAP-style attributions for a fine-tuned AlphaGenome head.

    The attribution target is the summed 1 bp output of one track (or the mean
    over all tracks of the head), optionally restricted to a centre window. For
    each sequence, the gradient w.r.t. the one-hot input minus the mean gradient
    over ``n_references`` reference sequences is returned. Prints the shapes of
    the generated references.

    Args:
        model: Loaded ``CustomAlphaGenomeModel`` (``alphagenome_ft.load_checkpoint``).
        sequence (jax.Array): One-hot DNA, shape ``(batch, seq_len, 4)``.
        organism_index (jax.Array): Organism index per sequence, shape ``(batch,)``
            (0 = human, 1 = mouse).
        head_name (str, optional): Custom head to attribute. Selected automatically
            if the model has a single custom head. Defaults to None.
        output_index (int, optional): Target track index (0-based). Defaults to None.
        output_key (str, optional): Target track name (requires ``track_names``);
            takes priority over ``output_index``. Defaults to None (mean over tracks).
        track_names (list of str, optional): Track names in head output order.
            Defaults to None.
        return_scaled (bool, optional): Use ``scaled_predictions_1bp`` instead of
            ``predictions_1bp``. Defaults to True.
        n_references (int, optional): Number of reference sequences. Defaults to 20.
        reference_type (str, optional): ``'dinucleotide_shuffle'``, ``'shuffle'``,
            ``'uniform'`` or ``'gc_match'``. Defaults to 'dinucleotide_shuffle'.
        random_state (int, optional): Random seed. Defaults to None.
        center_mask_size (int, optional): Width (bp) of the centre window whose
            summed output is attributed. Defaults to None (whole window).

    Returns:
        jax.Array: Attributions of shape ``(batch, seq_len, 4)``.

    Raises:
        ValueError: If the head or track cannot be resolved, or ``reference_type``
            is unknown.
    """
    # ------------------------------------------------------------------
    # Validate head_name
    # ------------------------------------------------------------------
    if head_name is None:
        if len(model._custom_heads) == 0:
            raise ValueError("No custom heads found.")
        elif len(model._custom_heads) == 1:
            head_name = model._custom_heads[0]
        else:
            raise ValueError(
                f"Multiple custom heads: {model._custom_heads}. Specify head_name."
            )
    elif head_name not in model._custom_heads:
        raise ValueError(
            f"Head '{head_name}' not found. Available: {model._custom_heads}"
        )

    # ------------------------------------------------------------------
    # Generate reference sequences
    # ------------------------------------------------------------------
    batch_size, seq_len, _ = sequence.shape
    references = []

    if random_state is not None:
        np.random.seed(random_state)

    seq_np = np.asarray(sequence)

    for b in range(batch_size):
        seq_batch = seq_np[b:b+1]  # (1, seq_len, 4)
        print(f"[batch {b}] seq_batch shape: {seq_batch.shape}")

        if reference_type == 'dinucleotide_shuffle':
            seed_b = None if random_state is None else random_state + b
            seq_torch = torch.tensor(seq_batch).permute(0, 2, 1)  # (1, 4, seq_len)
            print(f"[batch {b}] seq_torch shape (channels first): {seq_torch.shape}")

            shuffled = dinucleotide_shuffle(seq_torch, n=n_references, random_state=seed_b)
            print(f"[batch {b}] shuffled shape (raw from tangermeme): {shuffled.shape}")  # (1, n_refs, 4, seq_len)

            refs_np = shuffled.squeeze(0).permute(0, 2, 1).numpy()  # (n_refs, seq_len, 4)
            print(f"[batch {b}] refs_np shape (channels last): {refs_np.shape}")

            references.append(refs_np)  # <-- inside if, batch_refs never used

        else:
            batch_refs = []
            for r in range(n_references):
                if reference_type == 'shuffle':
                    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
                    seq_str = ''.join(
                        [base_map[np.argmax(seq_batch[0, i])] for i in range(seq_len)]
                    )
                    seq_list = list(seq_str)
                    for _ in range(seq_len // 2):
                        i = np.random.randint(0, len(seq_list) - 1)
                        seq_list[i], seq_list[i+1] = seq_list[i+1], seq_list[i]
                    ref_str = ''.join(seq_list)
                    ref_onehot = np.zeros((1, seq_len, 4), dtype=np.float32)
                    for i, base in enumerate(ref_str):
                        if base in base_map:
                            base_idx = list(base_map.keys())[
                                list(base_map.values()).index(base)
                            ]
                            ref_onehot[0, i, base_idx] = 1.0

                elif reference_type == 'uniform':
                    ref_onehot = np.random.multinomial(
                        1, [0.25, 0.25, 0.25, 0.25], size=(1, seq_len)
                    ).astype(np.float32)

                elif reference_type == 'gc_match':
                    gc_content = np.sum(seq_batch[0, :, [1, 2]]) / seq_len
                    ref_onehot = np.zeros((1, seq_len, 4), dtype=np.float32)
                    for i in range(seq_len):
                        if np.random.random() < gc_content:
                            ref_onehot[0, i, np.random.choice([1, 2])] = 1.0
                        else:
                            ref_onehot[0, i, np.random.choice([0, 3])] = 1.0
                else:
                    raise ValueError(f"Unknown reference_type: '{reference_type}'")

                batch_refs.append(ref_onehot)
                print(f"[batch {b}] ref {r+1}/{n_references} shape: {ref_onehot.shape}")

            refs_np = np.concatenate(batch_refs, axis=0)  # (n_refs, seq_len, 4)
            print(f"[batch {b}] concatenated batch_refs shape: {refs_np.shape}")
            references.append(refs_np)  # <-- inside else, symmetric with dinucleotide branch

    print(f"Total reference sets generated: {len(references)} (expected {batch_size})")
    print(f"Each reference set shape: {references[0].shape}")  # should be (n_refs, seq_len, 4)
    # ------------------------------------------------------------------
    # Strand reindexing boilerplate
    # ------------------------------------------------------------------
    negative_strand_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
    if hasattr(model._base_model, '_metadata'):
        first_org = list(model._base_model._metadata.keys())[0]
        strand_reindexing = jax.device_put(
            model._base_model._metadata[first_org].strand_reindexing
        )
    else:
        strand_reindexing = jnp.array([], dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Gradient computation per batch element
    # ------------------------------------------------------------------
    all_attributions = []

    for b in range(batch_size):
        orig_seq = sequence[b:b+1]           # (1, seq_len, 4)
        ref_seqs = jnp.array(references[b])  # (n_refs, seq_len, 4)

        def compute_output(seq, _b=b):  # _b=b freezes b in the closure
            # ---- forward pass ----------------------------------------
            predictions = model._predict(
                model._params,
                model._state,
                seq,
                organism_index[_b:_b+1],
                negative_strand_mask=negative_strand_mask[_b:_b+1],
                strand_reindexing=strand_reindexing,
            )

            # ---- retrieve raw head output ----------------------------
            if hasattr(predictions, '_custom'):
                output = predictions._custom.get(head_name)
            elif hasattr(predictions, 'get'):
                output = predictions.get(head_name)
            else:
                output = predictions[head_name] if head_name in predictions else None

            if output is None:
                raise ValueError(f"Output for head '{head_name}' not found.")

            # ---- select 1bp resolution key ---------------------------
            if not isinstance(output, dict):
                raise ValueError(
                    f"Expected head '{head_name}' to return a resolution dict "
                    f"but got {type(output)}. Available keys unknown."
                )

            resolution_key = _get_1bp_key(output, return_scaled)
            output = output[resolution_key]
            # output shape now: (1, positions_1bp, n_tracks)

            # ---- select target track ---------------------------------
            n_tracks = output.shape[-1]
            track_idx = _resolve_track_index(
                output_index, output_key, track_names, n_tracks
            )

            if track_idx is not None:
                output = output[..., track_idx]     # (1, positions_1bp)
            else:
                output = jnp.mean(output, axis=-1)  # (1, positions_1bp)

            # ---- reduce to scalar for jax.grad ----------------------
            if center_mask_size is not None:
                _, out_len = output.shape
                center = out_len // 2
                counts = jnp.sum(output[:, center-center_mask_size//2:center+center_mask_size//2])
            else:
                counts = jnp.sum(output)
            return counts

        grad_fn = jax.grad(compute_output)

        # Gradient for the original sequence
        orig_grad = grad_fn(orig_seq).squeeze(0)  # (seq_len, 4)

        # Average gradients across reference sequences
        ref_grads = []
        for r in range(n_references):
            ref_grad = grad_fn(ref_seqs[r:r+1]).squeeze(0)  # (seq_len, 4)
            ref_grads.append(ref_grad)

        ref_grad_mean = jnp.mean(jnp.stack(ref_grads), axis=0)  # (seq_len, 4)

        # DeepSHAP: difference between original and reference mean gradients
        attribution = orig_grad - ref_grad_mean  # (seq_len, 4)
        all_attributions.append(attribution)

    return jnp.stack(all_attributions)  # (batch, seq_len, 4)


def one_hot_encode_sequences(seqs: List[str]):
    """One-hot encodes DNA sequences of equal length (A, C, G, T; case-insensitive).

    Any other character (e.g. ``N``) becomes an all-zero row.

    Args:
        seqs (list of str): Sequences of identical length.

    Returns:
        jax.Array: Float32 array of shape ``(n_seqs, seq_len, 4)``.
    """
    seq_len = len(seqs[0])
    # We default to 4 (which jax.nn.one_hot will map to [0, 0, 0, 0] for 'N' or unknown bases)
    lookup = np.full(256, 4, dtype=np.int32)
    for char, idx in zip("ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
        lookup[ord(char)] = idx

    # Join strings, convert to ascii bytes, and read directly as an 8-bit integer array
    joined_bytes = "".join(seqs).encode('ascii')
    byte_array = np.frombuffer(joined_bytes, dtype=np.uint8)
    int_matrix = lookup[byte_array].reshape(len(seqs), seq_len)

    return jax.nn.one_hot(int_matrix, num_classes=4, dtype=jnp.float32) #automatically drops indices >= num_classes, turning 4 ('N') into zeros


def _gaussian_kernel_1d(sigma: float, truncate: float, device=torch.device('cpu'), dtype=torch.float32):
    """Generates a 1D Gaussian kernel replicating scipy's calculation."""
    radius = int(truncate * sigma + 0.5)
    kernel_len = 2 * radius + 1
    
    coords = torch.arange(kernel_len, dtype=dtype, device=device)
    coords -= radius
    
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    
    return kernel.view(1, 1, -1), radius


def normalize_attributions_by_local_activity(hyp_scores: torch.Tensor, sigma: float = 1280.0, truncate: float = 2.0) -> torch.Tensor:
    """Normalises attributions by their Gaussian-smoothed local activity.

    The activity track is the standard deviation across the four nucleotide
    channels per position. It is smoothed with a Gaussian kernel (reflect
    padding), clipped from below at its global 25th percentile, and the
    attributions are divided by it.

    Args:
        hyp_scores (torch.Tensor): Attributions of shape ``(n_seq, 4, seq_len)``.
        sigma (float, optional): Kernel standard deviation (bp). Defaults to 1280.0.
        truncate (float, optional): Kernel radius in standard deviations. Defaults to 2.0.

    Returns:
        torch.Tensor: Normalised attributions, same shape as ``hyp_scores``.

    Raises:
        ValueError: If ``hyp_scores`` is not 3D.
    """
    if hyp_scores.ndim != 3:
        raise ValueError(f"Expected 3D tensor of shape (n_seq, channels, seq_len), got {hyp_scores.ndim}D")
        
    # 1. Standard deviation across the NUCLEOTIDE dimension (which is dim=1)
    # Input: [4, 4, 16384] -> Output: [4, 16384]
    activity_track = hyp_scores.std(dim=1, unbiased=False)
    
    # 2. Get Gaussian kernel and radius
    kernel, radius = _gaussian_kernel_1d(sigma, truncate, device=hyp_scores.device, dtype=hyp_scores.dtype)
    
    # 3. Add the channel dimension back for Conv1D and apply 'reflect' padding
    # Input: [4, 16384] -> Reshaped: [4, 1, 16384] -> Padded: [4, 1, 21504]
    activity_reshaped = activity_track.unsqueeze(1) 
    padded_track = F.pad(activity_reshaped, (radius, radius), mode='reflect')
    
    # 4. Apply 1D convolution
    # Output: [4, 16384]
    scores_std_wide = F.conv1d(padded_track, kernel).squeeze(1)
    
    # 5. Clip to the global 25th percentile
    wide_clip = torch.quantile(scores_std_wide, 0.25)
    scores_std_wide = torch.clamp(scores_std_wide, min=wide_clip)
    
    # 6. Normalize the original scores
    # scores_std_wide is [4, 16384]. We unsqueeze(1) to make it [4, 1, 16384]
    # This allows it to broadcast correctly against your [4, 4, 16384] tensor
    normalized_scores = hyp_scores / scores_std_wide.unsqueeze(1)
        
    return normalized_scores
