from firedrake import *
import numpy as np
from mpi4py import MPI

def load_balance(mesh):
    for i in range(2):
        if isinstance(mesh._topology, firedrake.mesh.ExtrudedMeshTopology):
            if i == 0:
                horiz_elt = FiniteElement("DQ", quadrilateral, 0)
                vert_elt = FiniteElement("DG", interval, 0)
            else:
                horiz_elt = FiniteElement("CG", quadrilateral, 1)
                vert_elt = FiniteElement("CG", interval, 1)
            elt = TensorProductElement(horiz_elt, vert_elt)
            Z = FunctionSpace(mesh, elt)
        else:
            if i == 0:
                Z = FunctionSpace(mesh, "DG", 0)
            else:
                Z = FunctionSpace(mesh, "CG", 1)
        owned_dofs = Z.dof_dset.sizes[1]
        comm = Z.mesh().mpi_comm()
        min_owned_dofs = comm.allreduce(owned_dofs, op=MPI.MIN)
        mean_owned_dofs = np.mean(comm.allgather(owned_dofs))
        max_owned_dofs = comm.allreduce(owned_dofs, op=MPI.MAX)
        string = 'cells' if i == 0 else 'vertices'
        with np.errstate(divide='ignore'):
            warning(BLUE % ("Load balance %s: %i vs %i vs %i (%.3f, %.3f)" % (
                string, min_owned_dofs, mean_owned_dofs, max_owned_dofs,
                max_owned_dofs/mean_owned_dofs, max_owned_dofs/min_owned_dofs
            )))

def rebalance(dm, i):
    try:
        dm.rebalanceSharedPoints(useInitialGuess=False, parallel=False)
    except:
        warning("Vertex rebalancing in serial from scratch failed on level %i" % i)
    try:
        dm.rebalanceSharedPoints(useInitialGuess=True, parallel=True)
    except:
        warning("Vertex rebalancing from initial guess failed on level %i" % i)
