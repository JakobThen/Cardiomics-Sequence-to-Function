"""
AlphaGenome Targets Configuration
=================================

Builds the targets configuration consumed by ``alphagenome_ft``: one head per
data modality, listing its BigWig tracks together with their precomputed
non-zero means.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def infer_modality(folder_name: str) -> tuple[str, str]:
    """Maps a modality folder name to the AlphaGenome head ID and head kind.

    Args:
        folder_name (str): Folder name; must contain CutnTag (or CUT&TAG, CNT,
            C&T), ATAC or RNA (case-insensitive).

    Returns:
        tuple: ``(head_id, kind)``, e.g. ``("multiome_ATAC", "atac")``.

    Raises:
        ValueError: If the name does not match a recognised modality.
    """
    mod_upper = folder_name.upper()
    if any(tag in mod_upper for tag in ["CUTNTAG", "CUT&TAG", "CNT", "C&T"]):
        return "CutnTag", "atac"
    elif "ATAC" in mod_upper:
        return "multiome_ATAC", "atac"
    elif "RNA" in mod_upper:
        return "multiome_RNA", "rna_seq"
    raise ValueError(f"'{folder_name}' does not match any recognized modalities (CutnTag, ATAC, or RNA).")


def build_targets_config(data_dir, modality_folders, nonzero_means, norm_tag=None) -> dict:
    """Builds the AlphaGenome targets configuration from BigWig folders.

    For every modality folder, all ``*.bw`` files are listed in alphabetical
    order. If ``norm_tag`` is given, only files containing it are kept and the
    labels are cut at the tag (``ATAC_X_100M_norm`` -> ``ATAC_X``). Each track's
    non-zero mean is looked up in ``nonzero_means`` by path; both sides are compared
    as real paths, so relative, absolute and symlinked spellings all match. Track
    paths in the config are written as real paths (symlink targets).

    Args:
        data_dir (str or Path): Base directory containing the modality folders.
        modality_folders (list of str): Subfolders of ``data_dir``, one per modality.
        nonzero_means (pd.DataFrame): Table with ``full_path`` and
            ``nonzero_mean_canonical`` columns (output of
            ``00_compute_nonzero_means_per_bw.sh``).
        norm_tag (str, optional): Normalisation tag to filter files by; labels
            are cut at the tag. Defaults to None.

    Returns:
        dict: ``{"heads": [...]}`` with one predefined head per modality folder
        (keys ``dir_name, id, source, kind, targets``).

    Raises:
        ValueError: If a folder name does not match a recognised modality.
        AssertionError: If a BigWig file is missing from ``nonzero_means``.
    """
    data_dir = Path(data_dir)
    modality_configs = []

    # Look up means by real path, independent of how data_dir was spelled
    mean_by_path = {
        os.path.realpath(path): mean
        for path, mean in zip(nonzero_means["full_path"], nonzero_means["nonzero_mean_canonical"])
    }

    for mod in modality_folders:
        # Infer the correct ID and kind based on standardized naming conventions
        head_id, kind = infer_modality(mod)

        target_dir = data_dir / mod
        targets = []

        if target_dir.exists():
            # Glob all BigWigs and sort by case-sensitive file name: track order defines the
            # model's output order and must not depend on the platform (Windows paths sort
            # case-insensitively)
            for bw_file in sorted(target_dir.glob("*.bw"), key=lambda p: p.name):
                label = bw_file.name.replace(".bw", "")

                if norm_tag:
                    # Skip files that don't match our desired normalization tag
                    if norm_tag not in label:
                        continue
                    # Clean the label by cutting it at the tag, so e.g. "_100M" turns both
                    # "ATAC_X_100M_norm" and "RNA_X_minus_100M" into "ATAC_X" / "RNA_X_minus"
                    label = label.split(norm_tag)[0]

                real_path = os.path.realpath(bw_file)

                # Verify that we actually have a precomputed mean for this specific file to prevent silent downstream errors
                assert real_path in mean_by_path, f"Input .bw file {bw_file} not found in nonzero track means CSV."

                # Extract and round the mean
                nonzero_mean = round(float(mean_by_path[real_path]), 3)

                targets.append({
                    "path": real_path,
                    "label": label,
                    "nonzero_mean": nonzero_mean
                })
        else:
            print(f"Warning: Could not find directory {target_dir}")

        if not targets:
            print(f"Warning: No valid .bw files found in {target_dir}")

        # Construct the head dictionary expected by AlphaGenome
        # (alphagenome_ft requires "source"; "dir_name" is only informational)
        modality_configs.append({
            "dir_name": mod,
            "id": head_id,
            "source": "predefined",
            "kind": kind,
            "targets": targets
        })

    # Wrap the heads in the top-level configuration key
    return {"heads": modality_configs}


def normalize_targets_config(config: dict, resolve_paths: bool = True, check_exists: bool = True) -> dict:
    """Converts a targets configuration to the current format.

    Each head gets the keys ``dir_name, id, source, kind, targets`` in this order
    (``source`` defaults to ``"predefined"``, ``dir_name`` to the folder of its
    first track). Track order and all other keys are kept unchanged, because
    labels are matched to model outputs by position. Optionally replaces track
    paths by their symlink targets.

    Args:
        config (dict): Targets configuration (``{"heads": [...]}``), e.g. an older one.
        resolve_paths (bool, optional): Replace each track path by its real path
            (``os.path.realpath``). Defaults to True.
        check_exists (bool, optional): Raise if a (resolved) track file is missing.
            Defaults to True.

    Returns:
        dict: The converted configuration (the input is not modified).

    Raises:
        FileNotFoundError: If ``check_exists`` is set and a track file is missing.
    """
    heads = []
    for head in config["heads"]:
        targets = []
        for target in head.get("targets", []):
            target = dict(target)
            if resolve_paths:
                target["path"] = os.path.realpath(target["path"])
            if check_exists and not Path(target["path"]).is_file():
                raise FileNotFoundError(f"Track of head '{head.get('id')}' not found: {target['path']}")
            targets.append(target)

        first_path = head["targets"][0]["path"] if head.get("targets") else ""
        normalized = {
            "dir_name": head.get("dir_name") or Path(first_path).parent.name,
            "id": head["id"],
            "source": head.get("source", "predefined"),
            "kind": head.get("kind"),
        }
        # Keep optional alphagenome_ft overrides (resolutions, apply_squashing, ...)
        normalized.update({k: v for k, v in head.items() if k not in normalized and k != "targets"})
        normalized["targets"] = targets
        heads.append(normalized)
    return {**{k: v for k, v in config.items() if k != "heads"}, "heads": heads}


def write_targets_config(config: dict, out_path) -> None:
    """Writes a targets configuration to YAML, creating parent folders.

    Args:
        config (dict): Configuration from `build_targets_config`.
        out_path (str or Path): Output YAML path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # default_flow_style=False enforces clean, block-style YAML formatting
        # sort_keys=False preserves the logical order we defined above rather than forcing alphabetical keys
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
