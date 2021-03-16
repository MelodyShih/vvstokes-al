from firedrake import *
from firedrake.petsc import PETSc
from firedrake.mg.utils import get_level

Nx = Ny = Nz = 8
Lx = Ly = Lz = 4
nref = 1
distp = {"partition": True, 
         "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
basemesh = RectangleMesh(Nx, Ny, Lx, Ly, distribution_parameters=distp, quadrilateral=True)
basemh = MeshHierarchy(basemesh, nref)
mh = ExtrudedMeshHierarchy(basemh, height=Lz, base_layer=Nz)

k = 2
horiz_elt = FiniteElement("CG", quadrilateral, k)
vert_elt  = FiniteElement("CG", interval, k)
elt = VectorElement(TensorProductElement(horiz_elt, vert_elt))
V = FunctionSpace(mh[-1], elt)

u = Function(V)
u.interpolate(Expression("sin(x[0])"))
File("/scratch1/04841/tg841407/stokes_2021-03-15/vtk-mini/uf_level"+str(level)+".pvd").write(ulevel)


## visualize sym(grad)
for level in range(nref):
    Vlevel = FunctionSpace(mh[level], elt)
    ulevel = Function(Vlevel)
    inject(u, ulevel)

    File("/scratch1/04841/tg841407/stokes_2021-03-15/vtk-mini/uc_level"+str(level)+".pvd").write(ulevel)
