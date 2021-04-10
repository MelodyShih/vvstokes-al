from firedrake import *
from firedrake.petsc import PETSc

Nx = Ny = Nz = 8
Lx = Ly = Lz = 4
nref = 1
k = 2

#hex mesh
quadbasemesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
quadbasemh = MeshHierarchy(quadbasemesh, nref)
hexmh = ExtrudedMeshHierarchy(quadbasemh, height=Lz, base_layer=Nz)


horiz_elt = FiniteElement("CG", quadrilateral, k)
vert_elt  = FiniteElement("CG", interval, k)
elt = TensorProductElement(horiz_elt, vert_elt)

xhex = SpatialCoordinate(hexmh[-1])
Vhex = FunctionSpace(hexmh[-1], elt)
ufhex = Function(Vhex)
ufhex.interpolate(sin(2*pi*xhex[0]))
File("/scratch1/04841/tg841407/stokes_2021-03-15/vtk-mini/ufhex.pvd").write(ufhex)


Vchex = FunctionSpace(hexmh[0], elt)
uchex = Function(Vchex)
inject(ufhex, uchex)
File("/scratch1/04841/tg841407/stokes_2021-03-15/vtk-mini/uchex.pvd").write(uchex)

#tet mesh
tetbasemesh = BoxMesh(Nx, Ny, Nz, Lx, Ly, Lz)
tetmh = MeshHierarchy(tetbasemesh, nref)

elt = FiniteElement("CG", tetmh[-1].ufl_cell(), k)

xtet = SpatialCoordinate(tetmh[-1])
Vtet = FunctionSpace(tetmh[-1], elt)
uftet = Function(Vtet)
uftet.interpolate(sin(2*pi*xtet[0]))
File("/scratch1/04841/tg841407/stokes_2021-03-15/vtk-mini/uftet.pvd").write(uftet)

Vctet = FunctionSpace(tetmh[0], elt)
uctet = Function(Vctet)
inject(uftet, uctet)
File("/scratch1/04841/tg841407/stokes_2021-03-15/vtk-mini/uctet.pvd").write(uctet)

