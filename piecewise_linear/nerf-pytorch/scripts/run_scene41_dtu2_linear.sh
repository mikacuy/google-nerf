#!/bin/bash
#SBATCH --partition=orion --qos=normal
# #SBATCH --time=96:00:00  --> this is a comment, you can choose to not specify a nodelist, it will randomly assign to a GPU
#SBATCH --nodes=1
#SBATCH --nodelist=orion-dgx
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --account=orion

# only use the following on partition with GPUs
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=linear_41
#SBATCH --output=/orion/u/w4756677/slurm_dump/slurm-linear_41-%j.out

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
python run_nerf_constantinit.py --task train --config configs/dtu2_linear_improved.txt --num_train 40  --data_dir /orion/group/rs_dtu_4/DTU  --dtu_scene_id 41 --expname scan41_linear --i_img 10000
echo "Done"
