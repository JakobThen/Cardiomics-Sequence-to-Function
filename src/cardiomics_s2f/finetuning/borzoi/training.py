"""
Borzoi Training Helpers
=======================

Helpers for ``workflows/borzoi/finetune_borzoi.py``:

* :func:`find_normalized_bigwig_tracks` lists the normalised BigWigs in the staged
  ``*bigwig*`` folders (new runs).
* :func:`tracks_from_checkpoint` maps the task list stored in a checkpoint back to the
  staged files, in the checkpoint's order (resumed runs).
* :func:`warn_on_deduplicated_folder_pairs` warns when a folder and its
  ``*_deduplicated`` counterpart are staged together.
* :func:`unfreeze_embedding_layers` unfreezes selected layers of the Borzoi trunk.
* :func:`find_checkpoint_file` resolves a checkpoint file from a run folder.
* :func:`resolve_staged_track_paths` maps the node-local BigWig paths stored with saved
  predictions back to ``bigwig_dir``.

Track tables have the columns ``modality`` (folder name up to ``bigwig``, e.g.
``multiome_RNA_3prime``), ``file_name`` (used as the task name) and ``path``.
"""

import warnings
from pathlib import Path

import pandas as pd

TRACK_COLUMNS = ["modality", "file_name", "path"]


def _staged_bigwig_folders(data_dir, folders=None):
    """Returns the ``*bigwig*`` folders of ``data_dir`` (or the requested subset), sorted by name."""
    data_dir = Path(data_dir)
    if folders is None:
        found = [p for p in data_dir.iterdir() if p.is_dir() and "bigwig" in p.name]
    else:
        found = [data_dir / f for f in folders]
        missing = [str(p) for p in found if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"BigWig folders not found: {missing}")
    return sorted(found, key=lambda p: p.name)


def _modality_from_folder(folder_name):
    # split at "bigwig", take the first part, and strip the trailing underscore
    return folder_name.split("bigwig")[0].rstrip("_")


def warn_on_deduplicated_folder_pairs(folder_names):
    """Warns if a BigWig folder is staged together with its ``*_deduplicated`` version.

    The ``*_deduplicated`` folders (e.g. ``multiome_ATAC_bigwig_deduplicated``) hold
    files with the same names as their counterparts without the suffix, so a track
    name alone cannot tell which of the two a checkpoint was trained on.

    Args:
        folder_names (iterable of str): Names of the staged folders.

    Returns:
        list of tuple: ``(folder, folder_deduplicated)`` pairs found (empty if none).
    """
    names = set(folder_names)
    pairs = sorted(
        (name[: -len("_deduplicated")], name)
        for name in names
        if name.endswith("_deduplicated") and name[: -len("_deduplicated")] in names
    )
    for original, deduplicated in pairs:
        warnings.warn(
            f"Both '{original}' and '{deduplicated}' are staged. They contain files with the "
            "same names; keep only one of them in the input folder.",
            stacklevel=2,
        )
    return pairs


def find_normalized_bigwig_tracks(data_dir, folders=None, norm_tag="100M"):
    """Lists the normalised BigWig files in the staged BigWig folders.

    Args:
        data_dir (str or Path): Staged input directory (e.g. the node-local copy of
            ``seq2fun_train_input_baseres``).
        folders (list of str, optional): Folder names to use. Defaults to every folder
            of ``data_dir`` whose name contains ``bigwig``.
        norm_tag (str, optional): Only files whose name contains this tag are used.
            Defaults to "100M".

    Returns:
        pd.DataFrame: One row per track (columns ``modality``, ``file_name``, ``path``),
        sorted by folder name, then file name.

    Raises:
        ValueError: If no matching BigWig file is found.
    """
    data_dir = Path(data_dir)
    search_paths = _staged_bigwig_folders(data_dir, folders)
    warn_on_deduplicated_folder_pairs(p.name for p in search_paths)

    rows = []
    print(f"Scanning staged data directory: {data_dir}", flush=True)
    for search_path in search_paths:
        modality = _modality_from_folder(search_path.name)
        print(f"Found staged folder '{search_path.name}'. Extracted modality: '{modality}'", flush=True)

        for path in sorted(search_path.glob("*.bw"), key=lambda p: p.name):
            file_name = path.name
            if norm_tag not in file_name:
                continue  # use only normalized files
            rows.append({"modality": modality, "file_name": file_name, "path": str(path)})
    tracks = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    if tracks.empty:
        raise ValueError(f"No valid bigwig files found in {data_dir}. Check your staging logic.")
    print(f"Found {len(tracks)} .bw files, using them as input.", flush=True)
    return tracks


def tracks_from_checkpoint(model, data_dir, folders=None):
    """Maps the task names stored in a checkpoint to the staged BigWig files.

    The tasks of a gReLU checkpoint are the BigWig file names it was trained on
    (``model.data_params["tasks"]["name"]``), in output order. Resumed training must
    use the same files in the same order.

    Args:
        model: gReLU ``LightningModel`` loaded from the checkpoint.
        data_dir (str or Path): Staged input directory.
        folders (list of str, optional): Folder names to search. Defaults to every folder
            of ``data_dir`` whose name contains ``bigwig``.

    Returns:
        pd.DataFrame: One row per checkpoint task (columns ``modality``, ``file_name``,
        ``path``), in the checkpoint's task order.

    Raises:
        ValueError: If a task file is missing or found in more than one folder.
    """
    task_names = list(model.data_params["tasks"]["name"])
    search_paths = _staged_bigwig_folders(data_dir, folders)
    warn_on_deduplicated_folder_pairs(p.name for p in search_paths)

    candidates = {}
    for search_path in search_paths:
        modality = _modality_from_folder(search_path.name)
        for path in search_path.glob("*.bw"):
            candidates.setdefault(path.name, []).append(
                {"modality": modality, "file_name": path.name, "path": str(path)}
            )

    missing = [t for t in task_names if t not in candidates]
    ambiguous = {t: [c["path"] for c in candidates[t]] for t in task_names if len(candidates.get(t, [])) > 1}
    if missing or ambiguous:
        raise ValueError(
            f"Checkpoint tasks do not match the staged BigWigs in {data_dir}. "
            f"Missing: {missing}. Found in several folders: {ambiguous}. "
            "Remove the redundant folders or pass the folders to search."
        )
    print(f"Matched all {len(task_names)} checkpoint tasks to staged .bw files.", flush=True)
    return pd.DataFrame([candidates[t][0] for t in task_names], columns=TRACK_COLUMNS)


def unfreeze_embedding_layers(model, layer_substrings):
    """Unfreezes the Borzoi embedding (trunk) parameters whose names match.

    Every parameter of ``model.model.embedding`` whose name contains one of
    ``layer_substrings`` gets ``requires_grad = True``; all others are frozen. With an
    empty list the embedding is left unchanged.

    Args:
        model: gReLU ``LightningModel`` wrapping a Borzoi model.
        layer_substrings (list of str): Name parts to unfreeze, e.g.
            ``["unet_tower", "pointwise_conv"]`` (decoder) or ``["transformer_tower"]``.
    """
    print("\n--- Layer Freezing Configurations ---", flush=True)
    if layer_substrings:
        for name, param in model.model.embedding.named_parameters():
            if any(target in name for target in layer_substrings):
                param.requires_grad = True
                print(f"Unfrozen: {name}")
            else:
                param.requires_grad = False
    else:
        print("No layers specified to unfreeze. Embedding remains frozen.")


def find_checkpoint_file(path, pattern="epoch*.ckpt"):
    """Resolves the checkpoint file of a gReLU training run.

    gReLU (PyTorch Lightning) writes the best checkpoint as
    ``<save_dir>/<run name>/version_0/checkpoints/epoch=..-step=...ckpt`` next to
    ``last.ckpt``. A folder must contain exactly one file matching ``pattern``
    (searched recursively), so a folder with several runs is never resolved silently.

    Args:
        path (str or Path): A ``.ckpt`` file, or a folder containing one run.
        pattern (str, optional): File pattern of the checkpoint to use. Defaults to
            ``"epoch*.ckpt"`` (the best checkpoint, not ``last.ckpt``).

    Returns:
        Path: The checkpoint file.

    Raises:
        FileNotFoundError: If ``path`` does not exist or the folder has no match.
        ValueError: If the folder has more than one match.
    """
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    matches = sorted(path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No '{pattern}' checkpoint found under {path}")
    if len(matches) > 1:
        listing = "\n  ".join(str(m) for m in matches)
        raise ValueError(f"Several '{pattern}' checkpoints under {path}; pass one file instead:\n  {listing}")
    return matches[0]


def resolve_staged_track_paths(tracks_df, bigwig_dir):
    """Maps the staged BigWig paths of a training run back to ``bigwig_dir``.

    The track table saved with the test predictions holds the node-local copies
    (``<tmp>/<folder>/<file>``). The folder name is kept by the staging, so each track
    resolves exactly to ``<bigwig_dir>/<folder>/<file>``, also when several folders hold
    files with the same name.

    Args:
        tracks_df (pd.DataFrame): Track table with ``file_name`` and ``path`` columns, in
            prediction order (e.g. from
            :func:`~cardiomics_s2f.eval.track_prediction.read_h5_track_table`).
        bigwig_dir (str or Path): Folder containing the ``*bigwig*`` folders.

    Returns:
        pd.DataFrame: Copy of ``tracks_df`` with ``path`` pointing into ``bigwig_dir`` and
        the original value in ``staged_path``.

    Raises:
        FileNotFoundError: If a resolved file does not exist.
    """
    bigwig_dir = Path(bigwig_dir)
    out = tracks_df.copy()
    out["staged_path"] = out["path"]
    out["path"] = [
        str(bigwig_dir / Path(staged).parent.name / file_name)
        for staged, file_name in zip(out["staged_path"], out["file_name"])
    ]
    missing = [p for p in out["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} BigWig files not found under {bigwig_dir}: {missing}")
    return out
