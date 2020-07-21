#!bin/zsh
NREF=3
N=20
CASE=3
DISCR="cg"
echo "discretization: " $DISCR
echo "N = " $N ", NREF=" $N ", CASE=" $CASE 

echo "gamma = 0"
python3 stokes.py --solver-type almg --gamma    0 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

echo "gamma = 1"
python3 stokes.py --solver-type almg --gamma    1 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

echo "gamma = 10"
python3 stokes.py --solver-type almg --gamma   10 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

echo "gamma = 100"
python3 stokes.py --solver-type almg --gamma  100 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR

echo "gamma = 1000"
python3 stokes.py --solver-type almg --gamma 1000 --nref $NREF --dr $DR --k 2 --case $CASE --N $N --itref 0 --discretisation $DISCR
