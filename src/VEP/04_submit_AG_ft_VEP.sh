#!/bin/bash
#Code by Jakob Then
#SBATCH -J AG_ft_VEP		# job name
#SBATCH -N 1						# number of nodes
#SBATCH -p gpu-training
#SBATCH --qos=high
#SBATCH --gpus=8              
#SBATCH --cpus-per-gpu=8       
#SBATCH --mem-per-gpu=64G  
#SBATCH -C ("gpu=A100"|"gpu=H100"|"gpu=H200"|"gpu=B200")  
#SBATCH -t 2-00:00:00				# runtime limit (D-HH:MM:SS)
#SBATCH --output=/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft/logs/AG_ft_VEP_%j.out
#SBATCH --error=/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft/logs/AG_ft_VEP_%j.err

#script to run VEP using fintuned AG model

printf "%s\n" "" "----------------------------------------" "-----------Starting job date:-----------"
date
printf "%s\n" "" "----------------------------------------" "--------------Launched by:--------------"
whoami
printf "%s\n" "" "----------------------------------------" "-----------------JOB ID:----------------" "${SLURM_JOB_ID}"
printf "%s\n" "" "----------------------------------------" "-------------execution Node:------------" "${SLURMD_NODENAME}"
printf "%s\n" "" "----------------------------------------" ""

# Load conda
source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
source /g/steinmetz/projects/variant2function_project/env/alphagenome_ft/initiate_alphagenome_ft_env.sh

#Confirming correct JAX and CUDA setup
bash test_env_config.sh

#set cache for XLA tune (this chashes benchamrks on what computations run fastest on our setup and enables faster start up when using it again)
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft/xla_cache"

#Run VEP script
srun python 04_AG_ft_VEP.py 

conda deactivate
echo "VEP done!"
printf "%s\n" "" "----------------------------------------" "------------Ending job date:------------"
date
printf "%s\n" "" "----------------------------------------" ""