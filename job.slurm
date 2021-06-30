#!/bin/bash
#
#SBATCH --job-name=stokes
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH -p cs
#SBATCH --time=01:30:00


#SBATCH --qos=cpuplus
#SBATCH --reservation=fw18

#SBATCH --mem=180GB
#SBATCH --nodes=1
#SBATCH --tasks=24
##SBATCH --nodes=8
##SBATCH --tasks=192
##SBATCH --nodes=64
##SBATCH --tasks=1536
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=2


 
#module load mvapich2/intel/2.3.5
module load openmpi/intel/4.0.5
module load intel/19.1.2
module load python/intel/3.8.6

cd $HOME
mpiexec echo $MV2_COMM_WORLD_LOCAL_RANK
mpiexec echo $OMPI_COMM_WORLD_LOCAL_RANK
echo 'Start copying at'
date
mpiexec ./copyandunpack.sh
echo 'Finished copying at'
date
. /tmp/bin/firedrake/bin/activate

ldd /tmp/bin/petsc/intel/lib/libpetsc.so


export LD_LIBRARY_PATH=/tmp/bin/firedrake/lib:$LD_LIBRARY_PATH
export CACHEDIR=/tmp/fdcache
#export CACHEDIR=$HOME/localfdcache
export PYOP2_CACHE_DIR=$CACHEDIR
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$CACHEDIR
export XDG_CACHE_HOME=$CACHEDIR
export PETSC_DIR=/tmp/bin/petsc
export PETSC_ARCH=intel
#export LD_PRELOAD=/share/apps/mvapich2/2.3.5/intel/lib/libmpi.so:$LD_PRELOAD
#export MV2_CPU_BINDING_LEVEL=core
#export MV2_CPU_BINDING_POLICY=scatter
#export MV2_ENABLE_AFFINITY=0
#export MV2_HOMOGENEOUS_CLUSTER=1
export LD_PRELOAD=/share/apps/openmpi/4.0.5/intel/lib/libmpi.so:$LD_PRELOAD
export PYOP2_DEBUG=0
export PYOP2_BACKEND_COMPILER=icc
export OMP_NUM_THREADS=1
export UCX_WARN_UNUSED_ENV_VARS=n

#firedrake-clean


cd $HOME/vvstokes-melody
NREF=1
CASE=5
DISCR="cg"
DIM=3
echo "discretization: " $DISCR
echo "NREF=" $NREF ", CASE=" $CASE
echo "gamma = 10"
filename="${SLURM_JOB_ID}_Nodes_${SLURM_NNODES}_Tasks_${SLURM_NPROCS}_NREF_${NREF}"

exec_args="--solver-type almg --gamma 1 --nref $NREF --k 2 --case $CASE --itref 0 --discretisation cg --quad --linearization stressvel --large -log_view :${filename}_log_1.txt"
echo $exec_args
mpiexec $HOME/loadcache.sh
#mpiexec $HOME/clearcache.sh

annotate-output srun python3 vonmises_land_3d.py $exec_args >> logs/${filename}_1.txt
mpiexec $HOME/savecache.sh
