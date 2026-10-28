"""
Borzoi Variant Effect Prediction
================================

Library code for ``workflows/borzoi/run_vep.py`` (fine-tuned Borzoi via gReLU).

SNVs are predicted with gReLU's ``VariantDataset`` (REF and ALT). Indels and MNVs use
:class:`StitchedIndelDataset`, which follows the Borzoi stitching strategy
(Korsakova et al. 2025, bioRxiv): three alleles per variant are predicted and the left
and right halves of the shifted allele are stitched together, so REF and ALT bins stay
aligned on both sides of the variant.

Prediction arrays have shape ``(n_variants, 2, n_tracks, n_bins)`` with REF at index 0
and ALT at index 1. Scores:

* :func:`score_variants_whole_window`: ``SUM``, ``logSUM`` and ``L2`` over all bins.
* :func:`score_variants_masked_windows`: ``logSED`` and ``L2ED`` in a window centred on
  the variant (ATAC 501 bp, Cut&Tag 2001 bp) and, for RNA, over the exons of every gene
  overlapping the output window (strand-matched RNA tracks only).

The fine-tuned models are trained on squashed RNA targets, so their RNA predictions are
unsquashed with :func:`unsquash_rna_predictions` before scoring.

gReLU and einops are imported inside the functions that need them.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from cardiomics_s2f.io.data_input import SelectiveSquashTransform


def _split_overall_idx(idx: int, dims: tuple) -> tuple:
    """Helper to decode 1D PyTorch Dataset index into multi-dimensional indices."""
    seq_idx = idx // (dims[1] * dims[2])
    rem = idx % (dims[1] * dims[2])
    augment_idx = rem // dims[2]
    allele_idx = rem % dims[2]
    return seq_idx, augment_idx, allele_idx


class StitchedIndelDataset(Dataset):
    """Dataset for Borzoi-style stitched inference on indels and MNVs.

    Yields three alleles per variant (and augmentation) to allow stitching after
    prediction (:func:`stitch_indel_predictions`):

    * insertions: ``[REF, ALT_left, ALT_right]``;
    * deletions: ``[REF_left, REF_right, ALT]``;
    * MNVs: ``[REF_left, REF_right, ALT]`` (stitching is not needed, but this avoids the
      allele arithmetic of gReLU's ``VariantDataset``).

    The ``*_right`` alleles are cropped from a window shifted right by the length
    difference of the alleles.

    Args:
        variants (pd.DataFrame): Variants with columns ``chrom``, ``pos``, ``ref``, ``alt``.
        seq_len (int): Model input length.
        genome (str, optional): Genome to read the sequences from.
        rc (bool, optional): Add reverse-complemented copies. Defaults to False.
        max_seq_shift (int, optional): Maximum sequence shift for augmentation. Defaults to 0.
        seed (int, optional): Random seed for augmentation.
        augment_mode (str, optional): "serial" or "random". Defaults to "serial".
    """

    def __init__(
        self,
        variants: pd.DataFrame,
        seq_len: int,
        genome: Optional[str] = None,
        rc: bool = False,
        max_seq_shift: int = 0,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:
        from grelu.data.augment import Augmenter

        self.genome = genome
        self.seq_len = seq_len
        self.rc = rc
        self.max_seq_shift = max_seq_shift

        self.max_ref_len = variants.ref.str.len().max()
        self.max_alt_len = variants.alt.str.len().max()

        self._load_alleles(variants)
        self.n_alleles = 3         # INCREASED TO 3 ALLELES FOR STITCHING

        self._load_seqs(variants)
        self.n_seqs = self.seqs.shape[0]

        self.augmenter = Augmenter(
            rc=self.rc,
            max_seq_shift=self.max_seq_shift,
            seq_len=self.seq_len,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)

    def _load_alleles(self, variants: pd.DataFrame) -> None:
        from grelu.sequence.format import strings_to_indices

        self.ref_indices = [strings_to_indices([r])[0].tolist() for r in variants.ref]
        self.alt_indices = [strings_to_indices([a])[0].tolist() for a in variants.alt]

    def _load_seqs(self, variants: pd.DataFrame) -> None:
        from grelu.data.preprocess import check_chrom_ends
        from grelu.sequence.format import convert_input_type
        from grelu.sequence.utils import resize
        from grelu.variant import variants_to_intervals

        self.intervals = variants_to_intervals(variants, seq_len=self.seq_len)
        self.target_seq_len = self.seq_len + (2 * self.max_seq_shift)

        # Buffer needs to accommodate shifting the window to the right by up to max_delta
        max_delta = max(self.max_ref_len, self.max_alt_len)
        self.padded_fetch_len = self.target_seq_len + (2 * max_delta) + 10

        seqs = resize(self.intervals, seq_len=self.padded_fetch_len)
        check_chrom_ends(seqs, genome=self.genome)
        self.seqs = convert_input_type(seqs, "indices", genome=self.genome)

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented * self.n_alleles

    def __getitem__(self, idx: int) -> Tensor:
        from grelu.sequence.format import indices_to_one_hot

        seq_idx, augment_idx, allele_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_augmented, self.n_alleles)
        )

        seq = self.seqs[seq_idx].tolist()
        anchor_idx = self.padded_fetch_len // 2

        # Calculate size difference
        delta = len(self.alt_indices[seq_idx]) - len(self.ref_indices[seq_idx])

        # THE BORZOI SHIFT LOGIC
        if delta > 0:
            # INSERTION -> [REF, ALT_Left, ALT_Right]
            if allele_idx == 0:     # REF (Standard)
                insert_allele, shift = self.ref_indices[seq_idx], 0
            elif allele_idx == 1:   # ALT_Left (Standard)
                insert_allele, shift = self.alt_indices[seq_idx], 0
            else:                   # ALT_Right (Shift right by delta)
                insert_allele, shift = self.alt_indices[seq_idx], delta

        else:
            # DELETION -> [REF_Left, REF_Right, ALT]
            if allele_idx == 0:     # REF_Left (Standard)
                insert_allele, shift = self.ref_indices[seq_idx], 0
            elif allele_idx == 1:   # REF_Right (Shift right by abs(delta))
                insert_allele, shift = self.ref_indices[seq_idx], abs(delta)
            else:                   # ALT (Standard)
                insert_allele, shift = self.alt_indices[seq_idx], 0

        # Splice in the chosen allele
        ref_len = len(self.ref_indices[seq_idx])
        mutated_seq = seq[:anchor_idx] + insert_allele + seq[anchor_idx + ref_len:]

        # Crop the sequence, applying the rightward shift if needed
        crop_start = anchor_idx - (self.target_seq_len // 2) + shift
        crop_end = crop_start + self.target_seq_len

        cropped_seq = mutated_seq[crop_start:crop_end]

        if len(cropped_seq) != self.target_seq_len:
            raise RuntimeError(f"Crop mismatch! Expected {self.target_seq_len}, got {len(cropped_seq)}.")

        cropped_seq = np.array(cropped_seq)
        seq_aug = self.augmenter(seq=cropped_seq, idx=augment_idx)

        return indices_to_one_hot(seq_aug)


def filter_variants_near_chrom_ends(var_df: pd.DataFrame, chrom_sizes: dict, seq_len: int) -> pd.DataFrame:
    """Filters out variants located too close to chromosome boundaries.

    Args:
        var_df (pd.DataFrame): DataFrame containing variant 'chrom', 'pos', 'ref', and 'alt' columns.
        chrom_sizes (dict): Dictionary mapping chromosome names to their maximum lengths.
        seq_len (int): The sequence length the model requires.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only variants with safe distances from chromosome ends.
    """
    # We calculate the worst-case padding expansion to guarantee no sequence fetch will fail downstream.
    max_delta = max(var_df["ref"].str.len().max(), var_df["alt"].str.len().max())
    safe_distance = (seq_len // 2) + max_delta + 10

    # Attaching the max chromosome length to each row enables efficient vectorized boundary checking.
    var_df["chrom_len"] = var_df["chrom"].map(chrom_sizes)

    # We enforce strict coordinate boundaries because the model cannot reliably predict on truncated sequences.
    valid_bounds_mask = (var_df["pos"] > safe_distance) & (var_df["pos"] < (var_df["chrom_len"] - safe_distance))
    filtered_df = var_df[valid_bounds_mask].drop(columns=["chrom_len"]).reset_index(drop=True)

    # Reporting the exact exclusion numbers helps maintain a clear audit trail for the dataset size.
    original_count = len(var_df)
    filtered_count = len(filtered_df)
    dropped_count = original_count - filtered_count

    if original_count > 0:
        dropped_pct = (dropped_count / original_count) * 100
    else:
        dropped_pct = 0.0
    print(f"Dropped {dropped_count} variants ({dropped_pct:.2f}%) due to proximity to chromosome ends.", flush=True)

    return filtered_df


def average_indel_augmentations(preds, n_augmented, n_alleles=3, rc=True):
    """Averages the augmented predictions of a :class:`StitchedIndelDataset`.

    gReLU's ``predict_on_dataset`` does not reshape the output of custom datasets, so
    the flat ``(variants * augmentations * alleles)`` axis is split here, predictions
    of reverse-complemented sequences are flipped back, and augmentations are averaged.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions of shape
            ``(n_variants * n_augmented * n_alleles, n_tracks, n_bins)``.
        n_augmented (int): Augmentations per variant (``dataset.n_augmented``).
        n_alleles (int, optional): Alleles per variant (``dataset.n_alleles``). Defaults to 3.
        rc (bool, optional): Whether the dataset added reverse complements
            (``dataset.rc``); they form the second half of the augmentations. Defaults to True.

    Returns:
        np.ndarray: Predictions of shape ``(n_variants, n_alleles, n_tracks, n_bins)``.
    """
    import torch
    from einops import rearrange

    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    preds = rearrange(  # Reshape using einops (b n a) t l -> b n a t l
        preds,
        "(b n a) t l -> b n a t l",
        n=n_augmented,
        a=n_alleles,
    )

    # Flip predictions for reverse complemented sequences
    if rc and preds.shape[-1] > 1:
        half_aug = n_augmented // 2
        preds[:, half_aug:, ...] = np.flip(
            preds[:, half_aug:, ...], axis=-1
        )

    # Combine predictions for augmented sequences (mean aggregation manually)
    if n_augmented > 1:
        preds = np.mean(preds, axis=1)
    else:
        preds = preds[:, 0, ...]     # squeeze out 'n' dimension
    return preds


def stitch_indel_predictions(preds3: np.ndarray, var_df: pd.DataFrame) -> np.ndarray:
    """Stitches the three-allele predictions of indels into REF/ALT predictions.

    Insertions: ALT = left half of ``ALT_left`` + right half of ``ALT_right``.
    Deletions and MNVs: REF = left half of ``REF_left`` + right half of ``REF_right``.

    Args:
        preds3 (np.ndarray): Predictions of shape ``(n_variants, 3, n_tracks, n_bins)``.
        var_df (pd.DataFrame): The variants (``ref``, ``alt``) in the same order.

    Returns:
        np.ndarray: Predictions of shape ``(n_variants, 2, n_tracks, n_bins)`` with REF and ALT.
    """
    n_vars, _, n_tracks, n_bins = preds3.shape
    center_bin = n_bins // 2

    # Pre-allocate the final (n_vars, 2, n_tracks, n_bins) array
    stitched_preds = np.zeros((n_vars, 2, n_tracks, n_bins), dtype=preds3.dtype)

    for i in range(n_vars):
        ref_seq = var_df.iloc[i]['ref']
        alt_seq = var_df.iloc[i]['alt']
        delta = len(alt_seq) - len(ref_seq)

        if delta > 0:
            # INSERTION: preds3 contains [REF, ALT_Left, ALT_Right]
            # Standard REF is unchanged
            stitched_preds[i, 0] = preds3[i, 0]

            # Stitch ALT: Left half of ALT_Left + Right half of ALT_Right
            stitched_preds[i, 1, :, :center_bin] = preds3[i, 1, :, :center_bin]
            stitched_preds[i, 1, :, center_bin:] = preds3[i, 2, :, center_bin:]

        else:
            # DELETION (or Substitution where delta == 0):
            # preds3 contains [REF_Left, REF_Right, ALT]

            # Stitch REF: Left half of REF_Left + Right half of REF_Right
            stitched_preds[i, 0, :, :center_bin] = preds3[i, 0, :, :center_bin]
            stitched_preds[i, 0, :, center_bin:] = preds3[i, 1, :, center_bin:]

            # Standard ALT is unchanged
            stitched_preds[i, 1] = preds3[i, 2]

    return stitched_preds


def unsquash_rna_predictions(preds, track_names, task_axis=-2):
    """Converts RNA predictions from the squashed Borzoi scale back to counts.

    The fine-tuned Borzoi models were trained on RNA targets transformed with
    :class:`~cardiomics_s2f.io.data_input.SelectiveSquashTransform` (``"Borzoi"``); the
    transform is not stored in the checkpoint. RNA tracks are those whose name contains
    ``RNA`` (case-insensitive), matching the ``multiome_RNA_*`` folders used in training.

    Args:
        preds (np.ndarray): Predictions with the tracks on ``task_axis``.
        track_names (list of str): Track (task) names in output order.
        task_axis (int, optional): Axis of the tracks. Defaults to -2, as in
            ``(n_variants, 2, n_tracks, n_bins)``.

    Returns:
        np.ndarray: ``float32`` copy with the RNA tracks unsquashed; other tracks unchanged.
    """
    rna_idx = [i for i, name in enumerate(track_names) if "RNA" in name.upper()]
    unsquash = SelectiveSquashTransform(
        rna_task_indices=rna_idx, task_axis=task_axis, transform="Borzoi", inverse=True
    )
    return unsquash(preds)


def read_gtf_exons(gtf_path):
    """Reads the exons of protein-coding and processed-transcript genes from a GTF.

    Only ``chr*`` chromosomes without chrM and chrY are kept.

    Args:
        gtf_path (str or Path): GENCODE-style GTF file.

    Returns:
        pd.DataFrame: Exon rows with the GTF columns plus ``gene_name``, ``gene_type``,
        ``gene_id`` and ``transcript_id``.
    """
    gtf = pd.read_csv(gtf_path, sep="\t", comment="#", header=None)
    gtf.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    # Extract gene_name and gene_type directly with regex
    gtf["gene_name"] = gtf["attribute"].str.extract(r'gene_name\s+"([^"]+)"')
    gtf["gene_type"] = gtf["attribute"].str.extract(r'gene_type\s+"([^"]+)"')
    gtf["gene_id"] = gtf["attribute"].str.extract(r'gene_id\s+"([^"]+)"')
    gtf["transcript_id"] = gtf["attribute"].str.extract(r'transcript_id\s+"([^"]+)"')

    # filter for correct chromosomes and protein coding genes
    gtf = gtf[(gtf.chrom.str.contains("chr")) &
        (~gtf.chrom.str.contains("chrM|chrY")) &
        (gtf.gene_type.str.contains("protein_coding|processed_transcript"))
        ].drop_duplicates()
    exons_df = gtf[gtf["feature"] == "exon"].copy()
    return exons_df


def score_variants_whole_window(preds, var_df, track_names):
    """Scores variants over the full output window.

    ``SUM = sum(alt - ref)``, ``logSUM = sum(log2(1 + alt) - log2(1 + ref))`` and
    ``L2 = sqrt(sum((log2(1 + alt) - log2(1 + ref))**2))``, summed over all bins.

    Args:
        preds (np.ndarray): Stitched predictions of shape ``(n_variants, 2, n_tracks, n_bins)``.
        var_df (pd.DataFrame): Variants with a ``variant_id`` column, in prediction order.
        track_names (list of str): Track names in output order.

    Returns:
        pd.DataFrame: Long table with columns ``variant_id``, ``track``, ``SUM_score``,
        ``L2_score`` and ``logSUM_score``.
    """
    n_vars, _, n_tracks, _ = preds.shape

    variant_ids = var_df.variant_id.to_list()

    # Compute the scores (shape: n_vars x n_tracks)
    SUM_score = np.sum(preds[:, 1, :, :] - preds[:, 0, :, :], axis=2)
    logSUM_score = np.sum(np.log2(1 + preds[:, 1, :, :]) - np.log2(1 + preds[:, 0, :, :]), axis=2)
    L2_score = np.sqrt(np.sum((np.log2(1 + preds[:, 1, :, :]) - np.log2(1 + preds[:, 0, :, :])) ** 2, axis=2))

    # Repeat variant IDs for each track, and tile track names for each variant
    var_col = np.repeat(variant_ids, n_tracks)
    track_col = np.tile(track_names, n_vars)

    long_df = pd.DataFrame({
        'variant_id': var_col,
        'track': track_col,
        'SUM_score': SUM_score.flatten(),
        'L2_score': L2_score.flatten(),
        'logSUM_score': logSUM_score.flatten()
    })

    return long_df


def score_variants_masked_windows(preds, var_df, exon_df, track_names, label_len):
    """Scores variants in variant-centred windows (ATAC, Cut&Tag) and over exons (RNA).

    ``logSED = log2(sum(alt)) - log2(sum(ref))`` and
    ``L2ED = ||log2(alt + 1) - log2(ref + 1)||_2`` over the masked bins:

    * ATAC tracks: 501 bp around the variant; Cut&Tag tracks (``K4``, ``K27``,
      ``CUTNTAG``, ``CUT&TAG`` in the name): 2001 bp;
    * RNA tracks: the exon bins of each gene overlapping the output window, using only
      RNA tracks on the gene's strand (one row per variant, gene and track).

    Args:
        preds (np.ndarray): Stitched predictions of shape ``(n_variants, 2, n_tracks, n_bins)``.
        var_df (pd.DataFrame): Variants (``variant_id``, ``chrom``, ``pos``) in prediction
            order; ``start``/``end`` of the output window are added in place.
        exon_df (pd.DataFrame): Exons (``chrom``, ``start``, ``end``, ``strand``,
            ``gene_name``), e.g. from :func:`read_gtf_exons`.
        track_names (list of str): Track names in output order.
        label_len (int): Length of the output window in bp.

    Returns:
        pd.DataFrame: Long table with columns ``variant_id``, ``track``, ``assay_type``,
        ``gene``, ``L2ED_score`` and ``logSED_score``.
    """
    n_vars, _, n_tracks, n_bins = preds.shape
    bin_size = label_len // n_bins

    # precompute global log2fc
    ref_preds = preds[:, 0, :, :]
    alt_preds = preds[:, 1, :, :]
    log2fc = np.log2(alt_preds + 1) - np.log2(ref_preds + 1) #shape (n_vars, n_tracks, n_bins)

    #define assay types from track names and add strand info
    assay_type = []
    for t in track_names:
        if "ATAC" in t.upper(): assay_type.append("ATAC")
        elif "RNA" in t.upper(): assay_type.append("RNA")
        elif "K4" in t.upper() or "K27" in t.upper() or "CUTNTAG" in t.upper() or "CUT&TAG" in t.upper(): assay_type.append("Cut&Tag")
        else: assay_type.append("misc")

    # NOTE: the strand is guessed from substrings of the track name ('+', 'PLUS', 'FWD'
    # / '-', 'MINUS', 'REV'), checked in this order. This is fragile: e.g. a name with a
    # hyphen or containing 'REV' anywhere is read as minus strand. It matches the current
    # file names (RNA_<celltype>_{plus,minus}_100M.bw); check it when track names change.
    track_strand = []
    for t in track_names:
        t_upper = t.upper()
        if '+' in t_upper or 'PLUS' in t_upper or 'FWD' in t_upper: track_strand.append('+')
        elif '-' in t_upper or 'MINUS' in t_upper or 'REV' in t_upper: track_strand.append('-')
        else: track_strand.append(".") # Unstranded tracks (e.g. ATAC)

    track_meta = pd.DataFrame({"name": track_names, "assay_type": assay_type, "strand": track_strand})

    # Define variant-centric assays and their respective window sizes in nt
    variant_assays = {
        "ATAC": 501,
        "Cut&Tag": 2001
    }

    #add start and end inforamtion for variants to check exon overlap
    var_df.loc[:, "start"] = var_df["pos"] - 1 - label_len // 2
    var_df.loc[:, "end"] = var_df["pos"] - 1 + label_len // 2

    results = []

    # VARIANT-CENTRIC SCORING (Vectorized)
    for assay, mask_width_nt in variant_assays.items():
        tracks = track_meta[track_meta['assay_type'] == assay].index.tolist()
        epi_tracks = track_meta.loc[tracks, 'name'].tolist()

        if not tracks:
            continue

        start_bin = math.floor((label_len - mask_width_nt) / (2 * bin_size))
        end_bin = start_bin + math.ceil(mask_width_nt / bin_size)
        start_bin = max(0, start_bin)
        end_bin = min(n_bins, end_bin)

        masked_log2fc = log2fc[:, tracks, start_bin:end_bin]
        masked_l2 = np.linalg.norm(masked_log2fc, ord=2, axis=2)
        masked_logSED = np.log2(np.sum(alt_preds[:, tracks, start_bin:end_bin], axis=2)) - np.log2(np.sum(ref_preds[:, tracks, start_bin:end_bin], axis=2))

        for v_idx, var in var_df.iterrows():
            for t_idx, track_name in enumerate(epi_tracks):
                results.append({
                    "variant_id": var.variant_id,
                    "track": track_name,
                    "assay_type": assay,
                    "gene": pd.NA,
                    "L2ED_score": masked_l2[v_idx, t_idx],
                    "logSED_score": masked_logSED[v_idx, t_idx]
                })

    # EXON-CENTRIC SCORING (Iterative)
    if 'RNA' in track_meta['assay_type'].values:
        gene_bounds = exon_df.groupby('gene_name').agg({
            'chrom': 'first',
            'start': 'min',
            'end': 'max',
            'strand': 'first' # all exons for a gene have the same strand
        }).reset_index()

        for v_idx, var in var_df.iterrows():
            v_chrom = var.chrom
            v_start = var.start
            v_end = var.end

            overlapping_genes = gene_bounds[
                (gene_bounds['chrom'] == v_chrom) &
                (gene_bounds['start'] < v_end) &
                (gene_bounds['end'] > v_start)
            ]

            if overlapping_genes.empty:
                continue

            local_exons = exon_df[
                (exon_df['chrom'] == v_chrom) &
                (exon_df['gene_name'].isin(overlapping_genes['gene_name']))
            ]

            for _, gene_row in overlapping_genes.iterrows():
                gene_name = gene_row['gene_name']
                gene_strand = gene_row['strand']

                stranded_rna_tracks = track_meta[
                    (track_meta['assay_type'] == 'RNA') &
                    (track_meta['strand'] == gene_strand)
                ]

                if stranded_rna_tracks.empty:
                    continue

                rna_track_indices = stranded_rna_tracks.index.tolist()
                rna_track_names_stranded = stranded_rna_tracks['name'].tolist()

                exon_mask = np.zeros(n_bins, dtype=bool)
                gene_exons = local_exons[local_exons['gene_name'] == gene_name]

                for _, exon in gene_exons.iterrows():
                    e_start = max(v_start, exon['start'])
                    e_end = min(v_end, exon['end'])

                    if e_start < e_end:
                        start_bin = math.floor((e_start - v_start) / bin_size)
                        end_bin = math.ceil((e_end - v_start) / bin_size)

                        start_bin = max(0, min(n_bins, start_bin))
                        end_bin = max(0, min(n_bins, end_bin))

                        if start_bin < end_bin:
                            exon_mask[start_bin:end_bin] = True

                if not np.any(exon_mask):
                    continue

                masked_log2fc = log2fc[v_idx, rna_track_indices][:, exon_mask]
                l2_norms = np.linalg.norm(masked_log2fc, ord=2, axis=1)

                masked_ref_pred = ref_preds[v_idx, rna_track_indices][:, exon_mask]
                masked_alt_pred = alt_preds[v_idx, rna_track_indices][:, exon_mask]

                logSED_score = np.log2(np.sum(masked_alt_pred, axis = 1)) - np.log2(np.sum(masked_ref_pred, axis = 1))

                for t_idx, track_name in enumerate(rna_track_names_stranded):
                    results.append({
                        "variant_id": var.variant_id,
                        "track": track_name,
                        "assay_type": "RNA",
                        "gene": gene_name,
                        "L2ED_score": l2_norms[t_idx],
                        "logSED_score": logSED_score[t_idx]
                    })

    return pd.DataFrame(results)
