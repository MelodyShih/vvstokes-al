import numpy as np
from firedrake import *
from firedrake.preconditioners.asm import ASMPatchPC
from firedrake.mg.utils import *

def get_basemesh_nodes(W):
    pstart, pend = W.mesh()._topology_dm.getChart()
    section = W.dm.getDefaultSection()
    basemeshoff = {}
    basemeshdof = {}
    degree = W.ufl_element().degree()[1]
    nlayers = W.mesh().layers
    div = nlayers + (degree-1)*(nlayers-1)
    for p in range(pstart, pend):
        dof = section.getDof(p)
        off = section.getOffset(p)
        assert dof%div == 0
        basemeshoff[p] = off
        basemeshdof[p] = dof//div
    return basemeshoff, basemeshdof

class ASMHexStarPC(ASMPatchPC):
    '''Patch-based PC using Star of mesh entities implmented as an
    :class:`ASMPatchPC`.

    ASMStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the topological star of the mesh entity
    specified by `pc_star_construct_dim`.
    '''

    _prefix = "pc_hexstar_"

    def get_patches(self, V):
        mesh = V._mesh
        nlayers = mesh.layers
        mesh_dm = mesh._topology_dm

        # Obtain the topological entities to use to construct the stars
        depth = PETSc.Options().getInt(self.prefix+"construct_dim", default=0)

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = []
        for (i, W) in enumerate(V):
            V_local_ises_indices.append(V.dof_dset.local_ises[i].indices)

        
        basemeshoff = []
        basemeshdof = []
        dm = mesh._topology_dm

        for (i, W) in enumerate(V):
            boff, bdof = get_basemesh_nodes(W)
            basemeshoff.append(boff)
            basemeshdof.append(bdof)

        # Build index sets for the patches
        ises = []
        (start, end) = mesh_dm.getDepthStratum(depth)
        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                continue

            # Create point list from mesh DM
            pt_array, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
            for k in range(nlayers):
                indices = []
                # Get DoF indices for patch
                for (i, W) in enumerate(V):
                    section = W.dm.getDefaultSection()
                    degree = W.ufl_element().degree()[1]
                    for p in pt_array.tolist():
                        dof = basemeshdof[i][p]
                        if dof <= 0:
                            continue
                        off = basemeshoff[i][p]
                        if k == 0:
                            begin = off
                            end = off+dof+(degree-1)*dof
                        elif k == nlayers-1:
                            begin = off+k*degree*dof-(degree-1)*dof
                            end = off+dof+k*degree*dof
                        else:
                            begin = off+k*degree*dof-(degree-1)*dof
                            end = off+dof+k*degree*dof+(degree-1)*dof
                        W_indices = np.arange(W.value_size*begin, W.value_size*end, dtype='int32')
                        indices.extend(V_local_ises_indices[i][W_indices])
                iset = PETSc.IS().createGeneral(indices, comm=COMM_SELF)
                ises.append(iset)
        # u = Function(V)
        # out = File('/tmp/tmp.pvd')
        # out.write(u)
        # for i in range(len(ises)):
        #     for j in ises[i].array:
        #         u.dat.data[j//3, 0] = 1.
        #     out.write(u)
        #     for j in ises[i].array:
        #         u.dat.data[j//3, 0] = 0.
        return ises
