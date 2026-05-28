#!/bin/bash
conda activate borzoi
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX