#!/bin/bash
#SBATCH -J Au_NaCl
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -p hfacnormal04
#SBATCH --exclusive

ulimit -u 20000
module purge
module load mpi/intelmpi/2018.4.274
module list

export OMP_NUM_THREADS=2

mpirun -n 24 /public/home/bjiangch/cqfeng/software/lammps/build/lmp_mpi -in in.lmp >out
