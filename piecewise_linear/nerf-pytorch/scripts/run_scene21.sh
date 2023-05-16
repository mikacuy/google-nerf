#!/bin/bash
#SBATCH --partition=orion --qos=normal
# #SBATCH --time=96:00:00  --> this is a comment, you can choose to not specify a nodelist, it will randomly assign to a GPU
#SBATCH --nodes=1
#SBATCH --nodelist=oriong2
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --account=orion

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanxp:1
#SBATCH --job-name=scan21_40
#SBATCH --output=/orion/u/w4756677/slurm_dump/slurm-scan21_40-%j.out

# only use the following if yo####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALLu want email notification


# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
source ~/.bashrc
cd /orion/u/w4756677/nerf/google-nerf/piecewise_linear/nerf-pytorch
conda activate nerf
python run_nerf_constantinit.py --task train --config configs/dtu_linear_improved.txt --num_train 40  --data_dir /orion/group/mvs_training/dtu  --dtu_scene_id 21 --expname log_dtu_scan21_40views_linearimproved_constantinit_coarseinit2000_lr5e-4 --i_img 500
echo "Done"
