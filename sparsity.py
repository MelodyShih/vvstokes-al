from firedrake import *
from firedrake.mg.ufl_utils import coarsen
from firedrake.mg.utils import get_level

################################################################################ 
######### Prebuild sparsity ####################################################
################################################################################
# This should only take like half a second, but due to a bug in
# PyOP2/petsc/mpi/..., this suddenly takes ~half a minute once more than 10
# nodes or so are used.
def cache_sparsity(ZZ, VV, QQ):
    level = get_level(ZZ.ufl_domain())[1]
    for i in range(level+1):
        def build_sparsity(V0, V1, nest=None, block_sparse=None):
            # don't have to actually save it anywhere, cached internally by PyOP2
            op2.Sparsity((V0.dof_dset, V1.dof_dset),
                         (V0.cell_node_map(), V1.cell_node_map()),
                         iteration_regions=((op2.ALL, ), ) ,
                         nest=nest, block_sparse=block_sparse)
        build_sparsity(ZZ, ZZ, nest=True, block_sparse=True)
        build_sparsity(QQ, QQ, nest=False, block_sparse=False)
        build_sparsity(VV, VV, nest=False, block_sparse=False)
        VV = coarsen(VV, coarsen)
        ZZ = coarsen(ZZ, coarsen)
        QQ = coarsen(QQ, coarsen)

