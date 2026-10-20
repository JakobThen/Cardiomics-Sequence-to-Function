"""
AlphaGenome Targets Configuration
=================================

Builds the targets configuration consumed by ``alphagenome_ft``: one head per
data modality, listing its BigWig tracks together with their precomputed
non-zero means.
"""
from __future__ import annotations

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
    tag is stripped from their labels. Each track's non-zero mean is looked up
    in ``nonzero_means`` by its path ``f"{data_dir}/{folder}/{file}"``.

    Args:
        data_dir (str or Path): Base directory containing the modality folders.
        modality_folders (list of str): Subfolders of ``data_dir``, one per modality.
        nonzero_means (pd.DataFrame): Table with ``full_path`` and
            ``nonzero_mean_canonical`` columns (output of
            ``00_compute_nonzero_means_per_bw.sh``).
        norm_tag (str, optional): Normalisation tag to filter files by and strip
            from labels. Defaults to None.

    Returns:
        dict: ``{"heads": [...]}`` with one head per modality folder.

    Raises:
        ValueError: If a folder name does not match a recognised modality.
        AssertionError: If a BigWig file is missing from ``nonzero_means``.
    """
    data_dir = Path(data_dir)
    modality_configs = []

    for mod in modality_folders:
        # Infer the correct ID and kind based on standardized naming conventions
        head_id, kind = infer_modality(mod)

        target_dir = data_dir / mod
        targets = []

        if target_dir.exists():
            # Glob all BigWigs and sort alphabetically to ensure deterministic ordering in the YAML
            for bw_file in sorted(target_dir.glob("*.bw")):
                label = bw_file.name.replace(".bw", "")

                if norm_tag:
                    # Skip files that don't match our desired normalization tag
                    if norm_tag not in label:
                        continue
                    # Clean the label for the final config by stripping out the tag
                    label = label.replace(norm_tag, "")

                expected_path = f"{target_dir}/{bw_file.name}"

                # Verify that we actually have a precomputed mean for this specific file to prevent silent downstream errors
                match = nonzero_means.loc[nonzero_means["full_path"] == expected_path]
                assert not match.empty, f"Input .bw file {expected_path} not found in nonzero track means CSV."

                # Extract and round the mean
                nonzero_mean = match["nonzero_mean_canonical"].values[0]
                nonzero_mean = round(float(nonzero_mean), 3)

                targets.append({
                    "path": str(bw_file.resolve()),
                    "label": label,
                    "nonzero_mean": nonzero_mean
                })
        else:
            print(f"Warning: Could not find directory {target_dir}")

        if not targets:
            print(f"Warning: No valid .bw files found in {target_dir}")

        # Construct the head dictionary expected by AlphaGenome
        modality_configs.append({
            "dir_name": mod,
            "id": head_id,
            "kind": kind,
            "targets": targets
        })

    # Wrap the heads in the top-level configuration key
    return {"heads": modality_configs}


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
