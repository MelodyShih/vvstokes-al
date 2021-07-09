## Installation

* Get Firedrake with tinyasm: 

Install firedrake with tinyasm (as backend for ASMPatchPC) by following the instructions on [their website](https://firedrakeproject.org/download.html):
```
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --tinyasm
```

* Get the changes of the firedrake codes that is required for the vvstokes-al codes. 

Go to the firdrake repo and checkout to the branch: https://github.com/MelodyShih/firedrake
```
cd /path/to/firedrake//src/firedrake
git remote add test git@github.com:MelodyShih/firedrake.git
git fetch test
git checkout test/melody/vvstokes-al
```

* Clone this repository and all its submodules:
```
git clone --recursive git@github.com:MelodyShih/vvstokes-al.git
cd vvstokes-al/
```

* Activate the firedrake venv and install the alfi codes
```
source /path/to/firedrake/bin/activate
pip3 install -e alfi/
```

## Usage

The firedrake venv needs to be activated:
```
source /path/to/firedrake/bin/activate
```
Now you can run:
```
python3 multisinker.py --solver-type almg --gamma 10 --nref 3 --dr 1e4 --k 3 --case 4 --N 16 --dim 2 --asmbackend tinyasm --nsinker 24 --quad
```
