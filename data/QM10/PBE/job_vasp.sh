#!/bin/bash
#SBATCH -o slurm-%j.out
#SBATCH -p 28c_256g
#SBATCH -J std_vasp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
begin=`date +%s`
echo "======== Job starts at `date +'%Y-%m-%d %T'` ======== "

source ~/.bashrc
conda activate ase
module load  vasp/normal/5.4.4 intel

# mpirun -n $SLURM_NPROCS vasp_std | tee out
python3 compute.py
