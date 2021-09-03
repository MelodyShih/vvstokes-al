module load openmpi/intel/4.0.5
export MPIDIR=/share/apps/openmpi/4.0.5/intel/
export MPIBIN=${MPIDIR}bin
export CC=$MPIBIN/mpicc
export CXX=$MPIBIN/mpicxx
export FC=$MPIBIN/mpif90
export PARDIR="bin"
mkdir -p $PARDIR
cd $PARDIR
export WDIR=`pwd`
export PETSC_DIR=$WDIR/petsc
export PETSC_ARCH=intel
git clone https://github.com/firedrakeproject/petsc.git
cd $WDIR/petsc
curl -O -L -k https://github.com/eigenteam/eigen-git-mirror/archive/3.3.3.tar.gz
./configure PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH \
    CFLAGS="-xCORE-AVX512 -O2" \
    CXXFLAGS="-xCORE-AVX512 -O2" \
    FFLAGS="-xCORE-AVX512 -O2" \
    --download-blacs \
    --download-chaco \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-parmetis \
    --download-plapack \
    --download-ptscotch=1 \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-blacs=1 \
    --with-blaslapack-dir=/share/apps/intel/19.1.2/compilers_and_libraries/linux/mkl/ \
    --with-chaco=1 \
    --with-cxx-dialect=C++11 \
    --with-debugging=0 \
    --with-hdf5=1 \
    --download-hdf5=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.4/src/hdf5-1.10.4.tar.bz2 \
    --with-hypre=1 \
    --with-make-np=48 \
    --with-metis=1 \
    --with-mpi-dir=$MPIDIR \
    --with-mpi=1 \
    --with-mumps=1 \
    --with-packages-build-dir=/tmp/petsc-firedrake/$PETSC_ARCH \
    --with-parmetis=1 \
    --with-plapack=1 \
    --with-precision=double \
    --with-python-exec=python3 \
    --with-python=1 \
    --with-scalapack=1 \
    --with-scalar-type=real \
    --with-shared-libraries=1 \
    --with-spai=1 \
    --with-suitesparse=1 \
    --with-superlu=1 \
    --with-superlu_dist=1 \
    --with-x=0 \
    --with-pic \
    --with-logging=1 \
    --download-eigen=3.3.3.tar.gz
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH -j 18 all

cd $WDIR
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

sed -i 's|    log.info("Installing %s" % package)|    log.info("Installing %s" % package)\n    if package in ["petsc4py/", "h5py/"]:\n        shutil.rmtree(package + ".git", ignore_errors=True)|' firedrake-install
    if libspatialindex_changed:
        with directory("libspatialindex"):
            # Clean source directory
            check_call(["git", "reset", "--hard"])
            check_call(["git", "clean", "-f", "-x", "-d"])
            # Patch Makefile.am to skip building test
            check_call(["sed", "-i", "-e", "/^SUBDIRS/s/ test//", "Makefile.am"])
            # Build and install
            check_call(["cmake", "-DCMAKE_INSTALL_PREFIX=" + firedrake_env, "."])
            #check_call(["./autogen.sh"])
            #check_call(["./configure", "--prefix=" + firedrake_env,
            #            "--enable-shared", "--disable-static"])
            check_call(["make"])
            check_call(["make", "install"])
    else:
        log.info("No need to rebuild libspatialindex")

module load python/intel/3.8.6
unset PYTHONPATH
/share/apps/python/3.8.6/intel/bin/python3 firedrake-install --no-package-manager --honour-petsc-dir --disable-ssh \
    --mpicc $MPIBIN/mpicc \
    --mpicxx $MPIBIN/mpicxx \
    --mpif90 $MPIBIN/mpif90 \
    --mpiexec $MPIBIN/mpiexec

export LD_LIBRARY_PATH=/tmp/bin/firedrake/lib:$LD_LIBRARY_PATH


source $WDIR/firedrake/bin/activate
cd $WDIR/firedrake/src/firedrake
git remote add melody git@github.com:MelodyShih/firedrake.git
git fetch melody
git checkout -b vvstokes
git merge melody/melody/stressvel-solver

sed -i 's/if L is 0:/if L == 0:/g' /tmp/bin/firedrake/src/firedrake/firedrake/variational_solver.py
sed -i 's/"-llapack"/ /g' firedrake/slate/slac/compiler.py
cd $HOME/vvstokes-hdiv/
pip3 install -vvv -e alfi/

cd $WDIR/firedrake/bin
curl -O https://gist.githubusercontent.com/jacksoncage/54743cf17121f6bc5b8eb3013dfd06ac/raw/37e208f758069a7b8a25e87ec71ef9bc0c03d8f5/annotate-output
chmod u+x annotate-output

cd $WDIR/firedrake/src
git clone --recursive git@github.com:florianwechsung/TinyASM.git
pip3 install -vvv -e TinyASM/

export ZIPNAME=firedrake-2021-06-29v3
cd $WDIR/..
zip -r $ZIPNAME-withsrc.zip $PARDIR/
cp $ZIPNAME-withsrc.zip $HOME

rm -rf $PETSC_DIR/.git
rm -rf $PETSC_DIR/src
rm -rf $PETSC_DIR/$PETSC_ARCH/obj
for pkg in COFFEE  fiat  FInAT  firedrake  h5py  libspatialindex  libsupermesh  loopy  petsc4py  pyadjoint  PyOP2  tsfc  ufl; do
    rm -rf $WDIR/firedrake/src/$pkg/.git
done
rm -rf $WDIR/firedrake/src/firedrake/.git
rm -rf $WDIR/firedrake/src/firedrake/.git
cd $WDIR/..
zip -r $ZIPNAME-withoutsrc.zip $PARDIR/
cp $ZIPNAME-withoutsrc.zip $HOME


cp $HOME/$ZIPNAME-withsrc.zip /tmp
unzip /tmp/$ZIPNAME-withsrc.zip -d /tmp



# make this change to petsc when running with pc_mg_log to avoid creation of the MG Apply stage

    diff --git a/src/ksp/pc/impls/mg/mg.c b/src/ksp/pc/impls/mg/mg.c
    index 2abb32e112..88f4a36443 100644
    --- a/src/ksp/pc/impls/mg/mg.c
    +++ b/src/ksp/pc/impls/mg/mg.c
    @@ -746,22 +746,22 @@ PetscErrorCode PCSetFromOptions_MG(PetscOptionItems *PetscOptionsObject,PC pc)
         }
     
     #if defined(PETSC_USE_LOG)
    -    {
    -      const char    *sname = "MG Apply";
    -      PetscStageLog stageLog;
    -      PetscInt      st;
    -
    -      ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
    -      for (st = 0; st < stageLog->numStages; ++st) {
    -        PetscBool same;
    -
    -        ierr = PetscStrcmp(stageLog->stageInfo[st].name, sname, &same);CHKERRQ(ierr);
    -        if (same) mg->stageApply = st;
    -      }
    -      if (!mg->stageApply) {
    -        ierr = PetscLogStageRegister(sname, &mg->stageApply);CHKERRQ(ierr);
    -      }
    -    }
    +    /*{*/
    +    /*  const char    *sname = "MG Apply";*/
    +    /*  PetscStageLog stageLog;*/
    +    /*  PetscInt      st;*/
    +
    +    /*  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);*/
    +    /*  for (st = 0; st < stageLog->numStages; ++st) {*/
    +    /*    PetscBool same;*/
    +
    +    /*    ierr = PetscStrcmp(stageLog->stageInfo[st].name, sname, &same);CHKERRQ(ierr);*/
    +    /*    if (same) mg->stageApply = st;*/
    +    /*  }*/
    +    /*  if (!mg->stageApply) {*/
    +    /*    ierr = PetscLogStageRegister(sname, &mg->stageApply);CHKERRQ(ierr);*/
    +    /*  }*/
    +    /*}*/
     #endif
       }
       ierr = PetscOptionsTail();CHKERRQ(ierr);

