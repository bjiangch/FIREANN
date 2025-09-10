#!/bin/bash
#SBATCH -J Au_response
#SBATCH -p hgpu8
#SBATCH -n 8
#SBATCH -t 240:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=%J.err
#SBATCH --output=%J.out

module load anaconda3/2020.07
module load cuda/12.1.1
# module load cuda/11.3.0
module load gcc/10.2.0
# conda environment
conda_env=pytorch_2.1.2_cuda_12.1
# conda_env=pytorch-1.12.1-gpu-11.3
# conda_env=m3gnet
echo Begin Time: `date`
echo Dirretory is: $PWD
echo This job run on the nodes:
cat $SLURM_JOB_NODELIST

#path to save the code
# path="/data1/home/hhu17/cqfeng/response/FIREANN-Response/"
path="/data1/home/hhu17/cqfeng/response/FIREANN-Response-assmbe/"

#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
WORLD_SIZE=`expr SLURMJOBNUMNODES\*SLURM_JOB_NUM_NODES \* NPROC_PER_NODE`

COMMAND="$path"
source activate $conda_env 

cd $SLURM_SUBMIT_DIR 
# nvcc -V > out
python3 -m torch.distributed.run --nproc_per_node=NPROCPERNODE−−maxrestarts=0−−nnodes=NPROC_PER_NODE --max_restarts=0 --nnodes=SLURM_JOB_NUM_NODES --standalone $COMMAND > out
# nvidia-smi > out
echo End Time: `date`
