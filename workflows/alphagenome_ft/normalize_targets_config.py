"""Converts an (older) AlphaGenome targets config to the current format.

Adds ``dir_name`` and ``source`` where missing, keeps the track order, and
replaces every track path by its symlink target. Run it on the cluster, where
the paths exist, e.g.:

    python workflows/alphagenome_ft/normalize_targets_config.py \
        --input /path/to/old_configs/simple_merged_deduplicated.yaml \
        --output workflows/alphagenome_ft/configs/simple_merged_deduplicated.yaml

Prints a per-head summary (track count and labels) to compare with the checkpoint.
"""
from __future__ import annotations

import argparse

import yaml

from cardiomics_s2f.preprocessing.ag_targets_config import normalize_targets_config, write_targets_config


def get_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the config converter.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(description="Convert a targets config to the current format")
    parser.add_argument("--input", type=str, required=True, help="Existing targets config (YAML)")
    parser.add_argument("--output", type=str, required=True, help="Converted targets config (YAML)")
    parser.add_argument("--keep_symlinks", action="store_true", help="Do not resolve track paths to their symlink targets")
    parser.add_argument("--no_check", action="store_true", help="Do not check that every track file exists")
    return parser


def main(args: argparse.Namespace) -> None:
    """Converts the config, writes it and prints a per-head summary.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    with open(args.input) as f:
        config = yaml.safe_load(f)

    converted = normalize_targets_config(
        config, resolve_paths=not args.keep_symlinks, check_exists=not args.no_check
    )
    write_targets_config(converted, args.output)

    for head in converted["heads"]:
        labels = [t.get("label", "") for t in head["targets"]]
        print(f"{head['id']} ({head['kind']}, {head['dir_name']}): {len(labels)} tracks")
        print("   " + ", ".join(labels))
    print(f"Written: {args.output}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
