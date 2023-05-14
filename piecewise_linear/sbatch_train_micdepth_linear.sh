#!/bin/bash
#SBATCH --partition=orion --qos=normal
# #SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=oriong10

#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

# only use the following on partition with GPUs
# SBATCH --gres=gpu:1

# SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:titanrtx:1
# SBATCH --gres=gpu:a5000:1
# SBATCH --gres=gpu:titanxp:1

#SBATCH --account=orion

#SBATCH --job-name=8-mic
#SBATCH --output=slurm_output/8-mic-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

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

# srun slurm_scripts/evaluate_000.sh
# srun slurm_scripts/evaluate_001.sh
# srun slurm_scripts/evaluate_002.sh
# srun slurm_scripts/evaluate_003.sh
# srun slurm_scripts/evaluate_004.sh
# srun slurm_scripts/evaluate_005.sh
# srun slurm_scripts/evaluate_006.sh
# srun slurm_scripts/evaluate_007.sh
srun slurm_scripts/evaluate_008.sh
# srun slurm_scripts/evaluate_009.sh
# srun slurm_scripts/evaluate_010.sh
# srun slurm_scripts/evaluate_011.sh
# srun slurm_scripts/evaluate_012.sh

# done
echo "Done"
