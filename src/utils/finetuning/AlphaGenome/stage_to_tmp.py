"""Utility script for staging AlphaGenome data to a local compute node.

This script reads a master YAML configuration file, extracts the paths of all 
required BigWig files, and efficiently copies them to a local temporary directory 
(e.g., `/tmp` on a compute node) using `rsync`. It then generates a new temporary 
YAML configuration file with updated paths pointing to the locally staged data, 
which significantly reduces I/O bottlenecks during distributed training or evaluation.
"""
from __future__ import annotations

import yaml
import subprocess
import time
import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the staging script.
    
    This function is isolated to allow Sphinx extensions (`sphinxarg.ext`) 
    to auto-generate CLI documentation natively without executing the script.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(description="Stage AlphaGenome data to local tmp and update YAML config.")
    
    # Required arguments
    parser.add_argument("--master_yaml", type=str, required=True, 
                        help="Path to the input master YAML config")
    parser.add_argument("--tmp_dir", type=str, required=True, 
                        help="Path to the local tmp directory on the compute node")
    
    # Optional argument
    parser.add_argument("--out_yaml", type=str, default=None, 
                        help="Path for the output temporary YAML (defaults to <tmp_dir>/tmp_bw_config.yaml)")

    return parser


def main(args: argparse.Namespace) -> None:
    """Executes the data staging and configuration update pipeline.

    Reads the master YAML configuration to identify required BigWig files, 
    executes an `rsync` system call to transfer them to the specified temporary 
    directory, and writes a modified YAML configuration file referencing the 
    newly staged file paths.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing 
            paths for the master config, output config, and temporary directory.

    Raises:
        RuntimeError: If the underlying `rsync` subprocess fails to transfer 
            the files.
    """
    # Define paths based on arguments
    master_yaml_path = Path(args.master_yaml)
    tmp_dir = Path(args.tmp_dir)
    
    if args.out_yaml:
        tmp_yaml_path = Path(args.out_yaml)
    else:
        tmp_yaml_path = tmp_dir / "tmp_bw_config.yaml"

    print(f"Reading master config from: {master_yaml_path}")
    with open(master_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Create the /tmp directory and the parent directory for the output YAML if needed
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract unique paths and update dictionary
    files_to_copy = set()
    for head in config.get("heads", []):
        for target in head.get("targets", []):
            orig_path = Path(target["path"])
            files_to_copy.add(str(orig_path))
            
            # Update the target path in the dictionary to point to the new /tmp directory
            target["path"] = str(tmp_dir / orig_path.name)

    print(f"Staging {len(files_to_copy)} BigWig files to {tmp_dir} using rsync...")
    start_time = time.time()

    # 2. Construct and run the rsync command
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # --info=progress2: gives a clean, consolidated progress bar in your SLURM log
    rsync_cmd = ["rsync", "-a", "--info=progress2"] + list(files_to_copy) + [str(tmp_dir)]
    
    try:
        # Execute rsync
        subprocess.run(rsync_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Rsync transfer failed! Ensure paths are correct. Error: {e}")

    print(f"Data staging completed in {time.time() - start_time:.2f} seconds.")

    # 3. Write the new temporary YAML
    with open(tmp_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    print(f"Temporary config written to: {tmp_yaml_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)