#!bin/zsh
DR=$1
NSINKER=$2
NREF=3
N=16
CASE=4
DISCR="cg"
DR=$(printf "%.14f" $DR)
PROGRAM="../stokes.py" #"topleftblock.py"
DIM=2
SOLVER=almg
ORDER=3
echo "discretization: " $DISCR
echo "N = " $N ", NREF=" $NREF ", CASE=" $CASE ", DR=" $DR

set -x
echo "gamma = 0, cheb"
mpirun -n 4 python3 $PROGRAM --solver-type almgcheb --gamma    0 --nref $NREF --dr $DR --k $ORDER --case $CASE --N $N --itref 0 --discretisation $DISCR --dim $DIM --quad --nsinker $NSINKER| grep "Linear ns_ solve converged" 

echo "gamma = 0"
mpirun -n 4 python3 $PROGRAM --solver-type $SOLVER --gamma    0 --nref $NREF --dr $DR --k $ORDER --case $CASE --N $N --itref 0 --discretisation $DISCR --dim $DIM --quad --nsinker $NSINKER| grep "Linear ns_ solve converged" 

echo "gamma = 1"
mpirun -n 4 python3 $PROGRAM --solver-type $SOLVER --gamma    1 --nref $NREF --dr $DR --k $ORDER --case $CASE --N $N --itref 0 --discretisation $DISCR --dim $DIM --quad --nsinker $NSINKER| grep     "Linear ns_ solve converged"

echo "gamma = 10"
mpirun -n 4 python3 $PROGRAM --solver-type $SOLVER --gamma   10 --nref $NREF --dr $DR --k $ORDER --case $CASE --N $N --itref 0 --discretisation $DISCR --dim $DIM --quad --nsinker $NSINKER| grep     "Linear ns_ solve converged"

echo "gamma = 100"
mpirun -n 4 python3 $PROGRAM --solver-type $SOLVER --gamma  100 --nref $NREF --dr $DR --k $ORDER --case $CASE --N $N --itref 0 --discretisation $DISCR --dim $DIM --quad --nsinker $NSINKER| grep     "Linear ns_ solve converged"

echo "gamma = 1000"
mpirun -n 4 python3 $PROGRAM --solver-type $SOLVER --gamma 1000 --nref $NREF --dr $DR --k $ORDER --case $CASE --N $N --itref 0 --discretisation $DISCR --dim $DIM --quad --nsinker $NSINKER| grep     "Linear ns_ solve converged"
