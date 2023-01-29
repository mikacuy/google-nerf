#! /bin/bash
source ~/.bashrc

export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_HOME=/usr/local/cuda-11.3
conda activate ngp_pl


