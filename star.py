import numpy as np
from firedrake import *
from firedrake.preconditioners.asm import ASMPatchPC
from firedrake.mg.utils import *

class ASMStarPlusPC(ASMPatchPC):
    '''
    Slight modification of firedrake.ASMStarPC that guarantees that every MPI
    process has at least one patch on it.
    '''

    _prefix = "pc_star_"

    def get_patches(self, V):
        mesh = V._mesh
        mesh_dm = mesh._topology_dm

        # Obtain the topological entities to use to construct the stars
        depth = PETSc.Options().getInt(self.prefix+"construct_dim", default=0)

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = []
        for (i, W) in enumerate(V):
            V_local_ises_indices.append(V.dof_dset.local_ises[i].indices)

        # Build index sets for the patches
        ises = []
        (start, end) = mesh_dm.getDepthStratum(depth)
        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                # if we're at the last point, and haven't added any dofs so far, then build a star around a ghost dof, otherwise continue
                if seed < end-1 or len(ises) > 0:
                    continue

            # Create point list from mesh DM
            pt_array, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)

            # Get DoF indices for patch
            indices = []
            for (i, W) in enumerate(V):
                section = W.dm.getDefaultSection()
                for p in pt_array.tolist():
                    dof = section.getDof(p)
                    if dof <= 0:
                        continue
                    off = section.getOffset(p)
                    # Local indices within W
                    W_indices = numpy.arange(off*W.value_size, W.value_size * (off + dof), dtype='int32')
                    indices.extend(V_local_ises_indices[i][W_indices])
            iset = PETSc.IS().createGeneral(indices, comm=COMM_SELF)
            ises.append(iset)
        return ises

