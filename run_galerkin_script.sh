#!bin/zsh

NREF=5
N=1
CASE=3
DISCR="cg"
DR=1.0e8
DR=$(printf "%.14f" $DR)
PROGRAM="galerkin_vs_rediscretization.py"
echo "discretization: " $DISCR
echo "N = " $N ", NREF=" $NREF ", CASE=" $CASE ", DR=" $DR

set -x
echo "galerkin"
mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma    0 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR --galerkin | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma    1 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR --galerkin | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma   10 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR --galerkin | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma  100 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR --galerkin | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma 1000 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR --galerkin | grep "Linear firedrake_0_ solve"

echo "re-discretization"
mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma    0 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR  | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma    1 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR  | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma   10 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR  | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma  100 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR  | grep "Linear firedrake_0_ solve"

mpirun -n 1 python3 $PROGRAM --solver-type almg --gamma 1000 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR  | grep "Linear firedrake_0_ solve"
