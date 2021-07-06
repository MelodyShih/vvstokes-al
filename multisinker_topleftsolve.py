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
def mu_expr(meshl, level=0):
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
vvstokesprob.set_viscosity(mu_expr)

#--------------------------------------
# Setup right hand side
#--------------------------------------
# rhs
mesh = vvstokesprob.get_mesh()
V, Q = vvstokesprob.get_functionspace(mesh,info=True)
v = TestFunction(V)
rhsweak = -10 * (chi_n(mesh)-1)*v[1] * dx(degree=deg)

#--------------------------------------
# Setup weak form of the variable viscosity Stokes eq
#--------------------------------------
# Dirichlet boundary condition
bc_fun = vvstokesprob.create_dirichletbcsfun(mesh)
vvstokesprob.set_bcsfun(bc_fun)

dim  = vvstokesprob.dim 
quad = vvstokesprob.quad
V,Q  = vvstokesprob.get_functionspace(mesh)
bcs  = [DirichletBC(V, Constant((0.,) * dim), "on_boundary")]
if dim == 3 and quad:
    bcs += [DirichletBC(V, Constant((0., 0., 0.)), "top"),
            DirichletBC(V, Constant((0., 0., 0.)), "bottom")]

# Weak form of Stokes
F = vvstokesprob.get_weakform_A(mesh,bcs)
F += rhsweak


#--------------------------------------
# Setup firedrake's Linear variational problem (stores in vvstokesprob) 
#--------------------------------------
a = lhs(F)
l = rhs(F)

# create solution vector
sol_u = Function(V)

# set firedrake LinearVariationalProblem
vvstokesprob.set_linearvariationalproblem(a, l, sol_u, bcs)

#======================================
# Setup VariableViscosityStokesSolver  
#======================================
vvstokessolver = VariableViscosityStokesSolver(vvstokesprob, 
                                      args.solver_type, 
                                      args.case,
                                      args.gamma,
                                      args.asmbackend)

common = {
    "snes_type": "ksponly",
    "ksp_type": "fgmres",
    "ksp_gmres_restart": 500,
    "ksp_norm_type": "unpreconditioned",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 500,
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
}

solver_lu = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist",
}

solver_hypre = {
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    #"pc_hypre_boomeramg_strong_threshold": 0.8,
    "pc_hypre_boomeramg_max_iter": 5,
    "pc_hypre_boomeramg_grid_sweeps_down": 5,
    "pc_hypre_boomeramg_grid_sweeps_up": 5,
    "pc_hypre_boomeramg_grid_sweeps_coarse": 1,
    "pc_hypre_boomeramg_cycle_type": "W",
    "pc_hypre_boomeramg_max_levels": 15, 
    #"pc_hypre_boomeramg_max_coarse_size": 1500,
    "pc_hypre_boomeramg_relax_type_coarse": "Gaussian-elimination",
    "pc_hypre_boomeramg_print_statistics": None,
    "ksp_view": None,
}

mg_levels_solver_rich = {
    "ksp_type": "fgmres",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
}

mg_levels_solver = {
    "ksp_type": "fgmres",
    "ksp_norm_type": "unpreconditioned",
    "ksp_max_it": 5,
    "pc_type": "python",
    "pc_python_type": "hexstar.ASMHexStarPC" if (dim==3 and quad==True) 
                                            else "firedrake.ASMStarPC",
    "pc_star_construct_dim": 0,
    "pc_star_backend": args.asmbackend,
    # "pc_star_sub_pc_asm_sub_mat_type": "seqaij",
    # "pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
    "pc_star_sub_sub_pc_factor_in_place": None,
    "pc_hexstar_construct_dim": 0,
    "pc_hexstar_backend": args.asmbackend,
    "pc_hexstar_sub_sub_pc_factor_in_place": None,
    # "pc_hexstar_sub_pc_asm_sub_mat_type": "seqaij",
    # "pc_hexstar_sub_sub_pc_factor_mat_solver_type": "umfpack",
}

solver_mg = {
    "pc_type": "mg",
    "pc_mg_type": "full",
    #"mg_levels": mg_levels_solver,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
}

if args.solver_type == "almg":
    params = {**common, **solver_mg}
    params["mg_levels"] = mg_levels_solver
elif args.solver_type == "almgrich":
    params = {**common, **solver_mg}
    params["mg_levels"] = mg_levels_solver_rich
elif args.solver_type == "allu":
    params = {**common, **solver_lu}
elif args.solver_type == "alamg":
    params = {**common, **solver_hypre}
else:
    raise ValueError("please specify almg, allu or alamg for --solver-type")

vvstokessolver.set_parameters(params)

def augtopleftblock_cb(X, J, ctx):
    mh, level = get_level(ctx._x.ufl_domain())
    BTWBlevel = vvstokessolver.BBCTWB_dict[level]
    if level == nref:
        rmap, cmap = J.getLGMap()
        J.axpy(1, BTWBlevel, J.Structure.SUBSET_NONZERO_PATTERN)
        J.setLGMap(rmap, cmap)
    else:
        rmap, cmap = J.getLGMap()
        J.axpy(1, BTWBlevel, J.Structure.SUBSET_NONZERO_PATTERN)
        J.setLGMap(rmap, cmap)

def modifyresidual_cb(X, F):
    if case == 4 or case == 5:
        F += BTWB*X
    else:
        return

# set firedrake LinearVariationalSolver
vvstokessolver.set_linearvariationalsolver(augtopleftblock=True,
                                         modifyresidual=False,
                                         augtopleftblock_cb=augtopleftblock_cb,
                                         modifyresidual_cb=modifyresidual_cb)
#solver = LinearVariationalSolver(vvstokesprob.lvproblem,
#                                 solver_parameters=params,
#                                 options_prefix="topleft_")
if vvstokessolver.solver_type == "almg" and vvstokesprob.discretisation == "cg":
    lvsolver = vvstokessolver.lvsolver
    dim  = vvstokesprob.dim
    quad = vvstokesprob.quad
    nref = vvstokesprob.nref
    gamma = vvstokessolver.gamma
    asmbackend = vvstokessolver.asmbackend
    def BTWBcb(level):
        return vvstokessolver.BBCTWB_dict[level]
    def Acb(level):
        ksp = lvsolver.snes.ksp
        ctx = get_appctx(ksp.pc.getMGSmoother(level).dm)
        A = ctx._jac
        A.form = ctx.J
        A.petscmat = A.petscmat
        return A
    V, Q = vvstokesprob.get_functionspace(mesh)
    tdim = vvstokesprob.mesh.topological_dimension()
    mu_transfer = mu_expr 
    vtransfer = AlgebraicSchoeberlTransfer((mu_transfer, gamma), 
                         Acb, BTWBcb, tdim, 'uniform',
                         backend=asmbackend,
                         hexmesh=(dim==3 and quad))
    qtransfer = NullTransfer()
    transfers = {V.ufl_element(): (vtransfer.prolong, 
                                   vtransfer.restrict, 
                                   inject),
                 Q.ufl_element(): (prolong, restrict, qtransfer.inject)}
    #vvstokessolver.set_transfers(transfers=transfers)

#======================================
# Solve the multisinker problem
#======================================
for i in range(args.itref+1):
    vvstokessolver.solve()
    #solver.solve()
    performance_info(COMM_WORLD, vvstokessolver)

#File("u.pvd").write(z.split()[0])
