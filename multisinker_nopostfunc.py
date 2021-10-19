'''
=======================================
Runs linear variable viscosity Stokes solver for a Problem with multiple 
sinkers.

Author:                Florian Wechsung
                       Melody Shih
=======================================
'''

from firedrake import *
from firedrake.petsc import PETSc
from alfi.transfer import *
from alfi import *
from functools import reduce
from firedrake.mg.utils import get_level
from balance import load_balance, rebalance

from VariableViscosityStokes import *
from Abstract import Vector
import copy 
import argparse
import numpy as np
PETSc.Sys.popErrorHandler()

import logging
#logging.basicConfig(level="INFO")

#======================================
# Parsing input arguments
#======================================

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--dr", type=float, default=1e8)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--case", type=int, default=3)
parser.add_argument("--nsinker", type=int, default=8)
parser.add_argument("--nonzero-rhs", dest="nonzero_rhs", default=False, 
                                                           action="store_true")
parser.add_argument("--nonzero-initial-guess", dest="nonzero_initial_guess", 
                                            default=False, action="store_true")
parser.add_argument("--quad", dest="quad", default=False, action="store_true")
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--w", type=float, default=0.0)
parser.add_argument("--discretisation", type=str, default="hdiv")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--quad-deg", type=int, dest="quad_deg", default=10)
parser.add_argument("--rebalance", dest="rebalance", default=False, action="store_true")
parser.add_argument("--asmbackend", type=str, choices=['tinyasm', 'petscasm'], 
                                                             default="tinyasm")
parser.add_argument("--perturbmesh", dest="perturbmesh", default=False, action="store_true")
args, _ = parser.parse_known_args()


nsinker = args.nsinker
nref = args.nref
dr = args.dr
k = args.k
N = args.N
case = args.case
w = args.w
gamma = Constant(args.gamma)
dim = args.dim
deg = args.quad_deg
divdegree = None
rebal = args.rebalance

#======================================
# Setup viscosity field mu(x)
#======================================
# radius of sinker
omega = 0.1 #0.4, 0.1 
# decay parameter
delta = 200 #10, 200
mu_min = Constant(dr**-0.5)
mu_max = Constant(dr**0.5)

def Max(a, b): return (a+b+abs(a-b))/Constant(2)
def chi_n(mesh):
    X = SpatialCoordinate(mesh)
    def indi(ci):
        return 1-exp(-delta * Max(0, sqrt(inner(ci-X, ci-X))-omega/2)**2)
    indis = []
    np.random.seed(1)
    for i in range(nsinker):
        if dim == 2:
            cx = 2+np.random.uniform(-1,1)
            cy = 2+np.random.uniform(-1,1)
            indis.append(indi(Constant((cx,cy))))
        elif dim == 3:
            cx = 2+np.random.uniform(-1,1)
            cy = 2+np.random.uniform(-1,1)
            cz = 2+np.random.uniform(-1,1)
            indis.append(indi(Constant((cx,cy,cz))))
        else:
            raise NotImplementedError("Only implemented for dim=2,3")
    return reduce(lambda x, y : x*y, indis, Constant(1.0))
def mu_expr(mesh,level=0):
    return (mu_max-mu_min)*(1-chi_n(mesh)) + mu_min

def mu(mesh):
    Qm = FunctionSpace(mesh, Q.ufl_element())
    return Function(Qm).interpolate(mu_expr(mesh))

#======================================
# Setup VariableViscosityStokesProblem
#======================================
vvstokesprob = VariableViscosityStokesProblem(dim, # dimension of the problem 
                                    args.quad, #triangular/quadrilateral mesh
                                    args.discretisation, # finite elems spaces
                                    k, # order of discretisation
                                    quaddegree=deg, #quadrature degree
                                    quaddivdegree=divdegree,
                                    memcheck=True) # qaudrature divdeg                      
# set basemesh, mesh hierarchy  
basemesh = vvstokesprob.create_basemesh("rectangle", N, N, N, 4, 4, 4)
vvstokesprob.set_meshhierarchy(basemesh, nref, rebal)

# perturb mesh
if args.perturbmesh:
    mh = vvstokesprob.get_meshhierarchy()
    Vc = VectorFunctionSpace(mh[0], "CG", 1)
    defo = Function(Vc)
    defodata = 2e-3 * np.random.standard_normal(size=mh[0].coordinates.dat.data[:, :].shape)
    defodata[mh[0].coordinates.function_space().boundary_nodes("on_boundary"), :] = 0.
    defo.dat.data[:] = defodata
    
    from firedrake.mg.utils import get_level
    for mesh in mh:
        _, level = get_level(mesh)
        PETSc.Sys.Print("level = %d\n" % level)
        if level > 0:
            Vf = VectorFunctionSpace(mesh, "CG", 1)
            defof = Function(Vf)
            firedrake.mg.interface.prolong(defo, defof)
            mesh.coordinates.dat.data[:, :] += defof.dat.data[:]
        else: 
            mesh.coordinates.dat.data[:, :] += defo.dat.data[:]

# set viscosity field
vvstokesprob.set_viscosity(mu_expr)

#--------------------------------------
# Setup right hand side
#--------------------------------------
# rhs
mesh = vvstokesprob.get_mesh()
assert k==2
if dim == 2:
    if args.quad:
        V  = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "CG", k-1)
    else:
        V  = VectorFunctionSpace(mesh, "CG", k)
        Q  = FunctionSpace(mesh, "DG", 0)
elif dim == 3:
    if args.quad:
        horiz_elt = FiniteElement("CG", quadrilateral, k)
        vert_elt = FiniteElement("CG", interval, k)
        elt = VectorElement(TensorProductElement(horiz_elt, vert_elt))
        V = FunctionSpace(mesh, elt)
        Q = FunctionSpace(mesh, "DQ", k-2)
    else:
        Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
        if k < 3:
            FB = FiniteElement("FacetBubble", mesh.ufl_cell(), 3)
            eleu = VectorElement(NodalEnrichedElement(Pk, FB))
        else:
            eleu = VectorElement(Pk)
        V = FunctionSpace(mesh, eleu)
        Q = FunctionSpace(mesh, "DG", 0)
else:
    raise NotImplementedError("Only implemented for dim=2,3")

Z = V*Q
from sparsity import cache_sparsity
cache_sparsity(Z, V, Q)

v, q = TestFunctions(Z)
rhsweak = -10 * (chi_n(mesh)-1)*v[1] * dx(degree=deg)
if args.nonzero_rhs:
    divrhs = SpatialCoordinate(mesh)[0]-2
else:
    divrhs = Constant(0)
rhsweak += divrhs * q * dx(degree=divdegree)

mh = vvstokesprob.get_meshhierarchy()
Vc, Qc = vvstokesprob.get_functionspace(mh[-1],info=True)

#--------------------------------------
# Setup weak form of the variable viscosity Stokes eq
#--------------------------------------
# Dirichlet boundary condition
bc_fun = vvstokesprob.create_dirichletbcsfun(mesh)
vvstokesprob.set_bcsfun(bc_fun)
bcs = vvstokesprob.get_bcs(mesh)
# Weak form of Stokes
F = vvstokesprob.get_weakform_stokes(mesh,bcs)
aug = vvstokesprob.get_weakform_augterm(mesh, args.gamma, 0.0)
F += aug
F += rhsweak


#--------------------------------------
# Setup firedrake's Linear variational problem (stores in vvstokesprob) 
#--------------------------------------
a = lhs(F)
l = rhs(F)

# create solution vector
sol_z = Function(Z)

# set firedrake LinearVariationalProblem
vvstokesprob.set_linearvariationalproblem(a, l, sol_z, bcs)

#======================================
# Setup VariableViscosityStokesSolver  
#======================================
vvstokessolver = VariableViscosityStokesSolver(vvstokesprob, 
                                      args.solver_type, 
                                      args.case,
                                      args.gamma,
                                      args.asmbackend, setBTWBdics=False)

# monitor residual
params = vvstokessolver.get_parameters()
params["ksp_monitor_true_residual"]=None
params["ksp_converged_reason"]=None

vvstokessolver.set_nsp()
# set firedrake LinearVariationalSolver
vvstokessolver.set_linearvariationalsolver(augtopleftblock=False, modifyresidual=False)
if args.solver_type == "almg":
    tdim = mesh.topological_dimension()
    mu_transfer = mu_expr
    vtransfer = PkP0SchoeberlTransfer((mu_transfer, gamma), tdim, "uniform", backend=args.asmbackend, b_matfree=True, hexmesh=(args.dim == 3 and args.quad))
    qtransfer = NullTransfer()
    transfers = {V.ufl_element(): (vtransfer.prolong, vtransfer.restrict, inject),
                 Q.ufl_element(): (prolong, restrict, qtransfer.inject)}
    vvstokessolver.set_transfers(transfers=transfers)
else:
    vvstokessolver.set_transfers(standard=True)

#======================================
# Solve the multisinker problem
#======================================
# Set initial value
if args.nonzero_initial_guess:
    sol_z.split()[0].project(Constant((1., 1.)))
    sol_z.split()[1].interpolate(SpatialCoordinate(mesh)[1]-2)

for i in range(args.itref+1):
    PETSc.Log.begin()
    vvstokessolver.solve()
    performance_info(COMM_WORLD, vvstokessolver)

#Citations.print_at_exit()
File("u.pvd").write(sol_z.split()[0])
