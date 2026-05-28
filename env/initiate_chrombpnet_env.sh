#!/bin/bash
#START WITH SOURCE
conda activate chrombpnet
#Setup PATHs to use Cuda 11 instead of 12
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX