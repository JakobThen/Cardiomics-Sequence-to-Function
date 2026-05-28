#!/bin/bash
# ==============================================================================
# Code by Jakob Then
# 
# Purpose: Computes the mean of non-zero track values for BigWig (.bw) files, 
#          specifically restricted to canonical chromosomes (chr1-22, X, Y). 
#          Designed for AlphaGenome finetuning data preparation.
# Input:   $1 - Directory containing input .bw files
# Output:  $2 - CSV file capturing basename, non-zero mean, and full path
# ==============================================================================

# SLURM Configuration
#SBATCH --job-name=bw_mean_calc
#SBATCH --output=logs/bw_mean_%j.log
#SBATCH --error=logs/bw_mean_%j.err
#SBATCH --time=04:00:00          # Adjust depending on the total number of BigWig files
#SBATCH --cpus-per-task=1        # Sequential processing per file
#SBATCH --mem=64G                # Allocated generously to load large chromosome arrays into memory safely
#SBATCH --partition=bigmem       # Ensure this matches your specific cluster topology

# Example usage:
# sbatch 00_compute_nonzero_means_per_bw.sh input_data/AlphaGenome_Borzoi_input input_data/metadata/bw_nonzero_means.csv

source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
#source env/initiate_alphagenome_ft_env.sh
source /g/steinmetz/projects/variant2function_project/env/alphagenome_ft//initiate_alphagenome_ft_env.sh

# Validate CLI arguments to prevent empty variable execution errors
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <input_directory> <output_file.csv>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_FILE="$2"

echo "Starting job on $(date)"
echo "Input directory: $INPUT_DIR"
echo "Output file: $OUTPUT_FILE"

echo "basename,nonzero_mean_canonical,full_path" > "$OUTPUT_FILE"

# Iterate through all BigWig files in the target directory
find "$INPUT_DIR" -type f -name "*.bw" | while read -r bw_file; do
    
    base_name=$(basename "$bw_file")
    
    # Calculate non-zero means using an inline Python script. 
    nonzero_mean=$(python3 -c '
import sys, pyBigWig, numpy as np

# Restrict analysis strictly to canonical chromosomes to prevent skewed means from alt-contigs or unplaced scaffolds
valid_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])

try:
    bw = pyBigWig.open(sys.argv[1])
    total_sum = 0
    total_count = 0
    
    for chrom, length in bw.chroms().items():
        if chrom not in valid_chroms:
            continue
            
        # Load chromosome values into a numpy array for fast, vectorized filtering
        vals = bw.values(chrom, 0, length, numpy=True)
        
        # Filter out zeroes and NaN values (unmapped regions)
        nz = vals[(vals != 0) & (~np.isnan(vals))]
        
        total_sum += nz.sum()
        total_count += len(nz)
        
    mean_val = total_sum / total_count if total_count > 0 else "NaN"
    print(mean_val)
    bw.close()
except Exception as e:
    print("Error_Reading_File")
' "$bw_file")

    # Append results incrementally to prevent data loss if the SLURM job times out unexpectedly
    echo "$base_name,$nonzero_mean,$bw_file" >> "$OUTPUT_FILE"
    echo "Processed: $base_name"
done

conda deactivate
echo "Job finished on $(date). Results saved to $OUTPUT_FILE"