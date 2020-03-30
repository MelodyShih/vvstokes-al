# Installation

First install firedrake by following the instructions on [their website](https://firedrakeproject.org/download.html):

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install

Now activate the firedrake venv

    source /path/to/firedrake/bin/activate

Then clone this repository and all its submodules:

    git clone --recursive git@github.com:florianwechsung/vvstokes-hdiv.git
    cd vvstokes-hdiv/

Now install the Block Jacobi code

    pip3 install -e matpatch/

# Usage

The firedrake venv needs to be activated:

    source /path/to/firedrake/bin/activate

Now you can run:

    python3 stokes.py --solver-type almg --gamma 1e2 --nref 1 --dr 1e4 --k 2
