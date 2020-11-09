'''
=======================================
Compare results between standard prolongation (only implmented for Qh=P0^disc)
and algebraic schöberl prolongation using multisinker problem with 
discretisation Pk-P0^disc 

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
from variablestokes import *

import copy 
import argparse
import numpy as np
from petsc4py import PETSc
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
parser.add_argument("--quad-deg", type=int, dest="quad_deg", default=20)
parser.add_argument("--rebalance", dest="rebalance", default=False, action="store_true")
parser.add_argument("--asmbackend", type=str, choices=['tinyasm', 'petscasm'], 
                                                             default="tinyasm")
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
def mu_expr(mesh):
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
                                    quaddivdegree=divdegree) # qaudrature divdeg                      
# set basemesh, mesh hierarchy  
basemesh = vvstokesprob.create_basemesh("rectangle", N, N, N, 4, 4, 4)
vvstokesprob.set_meshhierarchy(basemesh, nref)
# set viscosity field
vvstokesprob.set_viscosity(mu_expr, mu_max, mu_min)

#--------------------------------------
# Setup right hand side
#--------------------------------------
# rhs
mesh = vvstokesprob.get_mesh()
V, Q = vvstokesprob.get_functionspace(mesh,info=True)
Z = V*Q
v, q = TestFunctions(Z)
rhsweak = -10 * (chi_n(mesh)-1)*v[1] * dx(degree=deg)
if args.nonzero_rhs:
    divrhs = SpatialCoordinate(mesh)[0]-2
else:
    divrhs = Constant(0)
rhsweak += divrhs * q * dx(degree=divdegree)

#--------------------------------------
# Setup weak form of the variable viscosity Stokes eq
#--------------------------------------
# Dirichlet boundary condition
bcs = vvstokesprob.get_dirichletbcs(mesh)
# Weak form of Stokes
F = vvstokesprob.get_weakform_stokes(mesh,bcs)
F += rhsweak

#--------------------------------------
# Setup firedrake's Linear variational problem (stores in vvstokesprob) 
#--------------------------------------
a = lhs(F)
l = rhs(F)

# create solution vector
sol_z_case4 = Function(Z)
vvstokesprob.set_linearvariationalproblem(a, l, sol_z_case4, bcs)

# case 3:
# Add weak form of augmented term gamma*(div(u),div(v))*dx
sol_z_case3 = Function(Z)
vvstokesprob_c3 = copy.copy(vvstokesprob)
aug = vvstokesprob_c3.get_weakform_augterm(mesh, args.gamma, divrhs)
Fgamma = F + aug
a = lhs(F)
l = rhs(F)
vvstokesprob_c3.set_linearvariationalproblem(a, l, sol_z_case3, bcs)

#======================================
# Setup VariableViscosityStokesSolver  
#======================================
#--------------------------------------
# Case 4
#--------------------------------------
vvstokessolver = VariableViscosityStokesSolver(vvstokesprob, 
                                      args.solver_type, 
                                      4,
                                      args.gamma,
                                      args.asmbackend)
vvstokessolver.set_nsp()
vvstokessolver.set_linearvariationalsolver()


#--------------------------------------
# Case 3
#--------------------------------------
vvstokessolver_c3 = VariableViscosityStokesSolver(vvstokesprob_c3, 
                                                 args.solver_type, 
                                                 3,
                                                 args.gamma,
                                                 args.asmbackend,
                                                 setBTWBdics=False)
vvstokessolver_c3.set_nsp()
# Setup standard prolongation for PkP0 for case 3 "cg" discretisation
# (vvstokessolver_c3) 
vvstokessolver_c3.set_linearvariationalsolver(augtopleftblock=False,
                                              modifyresidual=False)
if vvstokesprob_c3.discretisation == "cg" and vvstokesprob_c3.discretisation == "almg":
    V = Z.sub(0)
    Q = Z.sub(1)
    tdim = mesh.topological_dimension()
    vtransfer = PkP0SchoeberlTransfer((mu_transfer, gamma), tdim, 
                hierarchy, backend=args.asmbackend, b_matfree=True, 
                hexmesh=(args.dim == 3 and args.quad))
    qtransfer = NullTransfer()
    transfers = {V.ufl_element(): 
                        (vtransfer.prolong,vtransfer.restrict,inject),
                 Q.ufl_element(): (prolong,restrict,qtransfer.inject)}
    vvstokessolver_c3.set_transfers(transfers=transfers)

#======================================
# Solve the multisinker problem
#======================================
# Set initial value
if args.nonzero_initial_guess:
    sol_z_case4.split()[0].project(Constant((1., 1.)))
    sol_z_case4.split()[0].project(Constant((1., 1.)))
    sol_z_case3.split()[1].interpolate(SpatialCoordinate(mesh)[1]-2)
    sol_z_case3.split()[1].interpolate(SpatialCoordinate(mesh)[1]-2)

for i in range(args.itref+1):
    PETSc.Log.begin()
    vvstokessolver.solve()
    performance_info(COMM_WORLD, vvstokessolver)

for i in range(args.itref+1):
    PETSc.Log.begin()
    vvstokessolver_c3.solve()
    performance_info(COMM_WORLD, vvstokessolver_c3)

PETSc.Sys.Print("absolute diff in vel:",\
       norm(sol_z_case4.split()[0]-sol_z_case3.split()[0]))
PETSc.Sys.Print("relative diff in vel:",\
       norm(sol_z_case4.split()[0]-sol_z_case3.split()[0])\
       /norm(sol_z_case3.split()[0]))
PETSc.Sys.Print("absolute diff in pre: ",\
       norm(sol_z_case4.split()[1]-sol_z_case3.split()[1]))
PETSc.Sys.Print("relative diff in pre: ",\
       norm(sol_z_case4.split()[1]-sol_z_case3.split()[1])\
       /norm(sol_z_case3.split()[1]))
