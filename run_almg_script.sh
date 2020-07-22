#!bin/zsh

NREF=3
N=35
CASE=3
DISCR="cg"
DR=1.0e6
DR=$(printf "%.14f" $DR)
PROGRAM="topleftblock.py"
echo "discretization: " $DISCR
echo "N = " $N ", NREF=" $NREF ", CASE=" $CASE ", DR=" $DR

set -x
#echo "gamma = 0"
mpirun -n  1 python3 $PROGRAM --solver-type almg --gamma    0 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR 

#echo "gamma = 1"
mpirun -n 16 python3 $PROGRAM --solver-type almg --gamma    1 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

#echo "gamma = 10"
mpirun -n 16 python3 $PROGRAM --solver-type almg --gamma   10 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

#echo "gamma = 100"
mpirun -n 16 python3 $PROGRAM --solver-type almg --gamma  100 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

#echo "gamma = 1000"
mpirun -n 16 python3 $PROGRAM --solver-type almg --gamma 1000 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR
