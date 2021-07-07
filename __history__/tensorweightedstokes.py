from firedrake import *
from firedrake.petsc import PETSc
from alfi.transfer import *
from functools import reduce
from firedrake.mg.utils import get_level

import argparse
import numpy as np
from petsc4py import PETSc

#import logging
#logging.basicConfig(level="INFO")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--dr", type=float, default=1e8)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--case", type=int, default=3)
parser.add_argument("--nonzero-rhs", dest="nonzero_rhs", default=False, action="store_true")
parser.add_argument("--nonzero-initial-guess", dest="nonzero_initial_guess", default=False, action="store_true")
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--w", type=float, default=0.0)
parser.add_argument("--discretisation", type=str, default="hdiv")
parser.add_argument("--dim", type=int, default=2)
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
N = args.N
case = args.case
w = args.w
gamma = Constant(args.gamma)
dim = args.dim

distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}


hierarchy = "uniform"
def before(dm, i):
     for p in range(*dm.getHeightStratum(1)):
         dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
     for p in range(*dm.getHeightStratum(1)):
         dm.setLabelValue("prolongation", p, i+2)

def mesh_hierarchy(hierarchy, nref, callbacks, distribution_parameters):
    if dim == 2:
        baseMesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp)
    elif dim == 3:
        baseMesh = BoxMesh(N, N, N, 4, 4, 4, distribution_parameters=distp)
    else:
        raise NotImplementedError("Only implemented for dim=2,3")

    if hierarchy == "uniform":
        mh = MeshHierarchy(baseMesh, nref, reorder=True, callbacks=callbacks,
                           distribution_parameters=distribution_parameters)
    else:
        raise NotImplementedError("Only know uniform for the hierarchy.")
    return mh
mh = mesh_hierarchy(hierarchy, nref, (before, after), distp)
#mh = MeshHierarchy(mesh, nref, reorder=True, distribution_parameters=distp)

mesh = mh[-1]

if args.discretisation == "cg":
    if dim == 2:
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "DG", 0)
    elif dim == 3:
        Pk = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
        FB = FiniteElement("FacetBubble", mesh.ufl_cell(), 3)
        eleu = VectorElement(NodalEnrichedElement(Pk, FB))
        V = FunctionSpace(mesh, eleu)
        Q = FunctionSpace(mesh, "DG", 0)
    else:
        raise NotImplementedError("Only implemented for dim=2,3")
else:
    raise ValueError("please specify hdiv or cg for --discretisation")


Z = V * Q
PETSc.Sys.Print("dim(Z) = ", Z.dim())
PETSc.Sys.Print("dim(V) = ", V.dim())
PETSc.Sys.Print("dim(Q) = ", Q.dim())
z = Function(Z)
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
if dim == 2:
    bcs = [DirichletBC(Z.sub(0), Constant((0., 0.)), "on_boundary")]
elif dim == 3:
    bcs = [DirichletBC(Z.sub(0), Constant((0., 0., 0.)), "on_boundary")]
else:
    raise NotImplementedError("Only implemented for dim=2,3")


omega = 0.1 #0.4, 0.1
delta = 200 #10, 200
mu_min = Constant(dr**-0.5)
mu_max = Constant(dr**0.5)

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

def chi_n(mesh):
    X = SpatialCoordinate(mesh)
    def indi(ci):
        return 1-exp(-delta * Max(0, sqrt(inner(ci-X, ci-X))-omega/2)**2)
    # indis = [indi(Constant((4*(cx+1)/3, 4*(cy+1)/3))) for cx in range(2) for cy in range(2)]
    indis = []
    np.random.seed(1)
    for i in range(8):
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
    # Another test:
    # for i in range(4):
    #     cx = 2+np.random.uniform(-1,1)
    #     cy = 2+np.random.uniform(-1,1)
    #     indis.append(indi(Constant((cx,cy))))
    # for i in range(2):
    #     cx = 3+np.random.uniform(-1,1)
    #     cy = 3+np.random.uniform(-1,1)
    #     indis.append(indi(Constant((cx,cy))))
    # for i in range(2):
    #     cx = 3+np.random.uniform(-1,1)
    #     cy = 1+np.random.uniform(-1,1)
    #     indis.append(indi(Constant((cx,cy))))
    # indis.append(indi(Constant((cx,cy))))

    return reduce(lambda x, y : x*y, indis, Constant(1.0))

def mu_expr(mesh):
    return (mu_max-mu_min)*(1-chi_n(mesh)) + mu_min

def mu(mesh):
    Qm = FunctionSpace(mesh, Q.ufl_element())
    return Function(Qm).interpolate(mu_expr(mesh))

#File("mu_"+str(N)+"_delta_"+str(delta)+".pvd").write(mu(mesh))
epsilon = 1.e-3
mumatrix = as_matrix(((mu_expr(mesh), Constant(0.0)),(Constant(0.0), Constant(epsilon))))
#mumatrix = as_matrix(((mu_expr(mesh), Constant(0.0)),(Constant(0.0), mu_expr(mesh))))
#mumatrix = as_matrix(((Constant(1.0), Constant(0.0)),(Constant(0.0), Constant(epsilon))))

sigma = Constant(100.)
if args.discretisation == "cg":
    h = CellDiameter(mesh)
else:
    raise ValueError("please specify hdiv or cg for --discretisation")
n = FacetNormal(mesh)

def diffusion(u, v, mu):
    #return (mu*inner(2*sym(grad(u)), grad(v)))*dx
    return (inner(2*mu*sym(grad(u)), sym(grad(v))))*dx

#F = diffusion(u, v, mu_expr(mesh))
F = diffusion(u, v, mumatrix)

if dim == 2:
    F += - p * div(v) * dx(degree=2*(k-1)) - div(u) * q * dx(degree=2*(k-1))
elif dim == 3:
    F += - p * div(v) * dx(degree=3*(k-1)) - div(u) * q * dx(degree=3*(k-1))

F += -10 * (chi_n(mesh)-1)*v[1] * dx
if args.nonzero_rhs:
    divrhs = SpatialCoordinate(mesh)[0]-2
else:
    divrhs = Constant(0)
F += divrhs * q * dx(degree=2*(k-1))

if args.discretisation == "cg":
    Fgamma = F + Constant(gamma)*inner(cell_avg(div(u))-divrhs, cell_avg(div(v)))*dx(degree=3*(k-1))
else:
    raise ValueError("please specify hdiv or cg for --discretisation")

if case == 3:
    a = lhs(Fgamma)
    l = rhs(Fgamma)
else:
    raise ValueError("Unknown type of preconditioner %i" % case)



fieldsplit_1 = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "schurcomplement.DGMassInv"
}


fieldsplit_0_lu = {
    "ksp_type": "preonly",
    "ksp_max_it": 1,
    "pc_type": "lu",
    #"pc_factor_mat_solver_type": "superlu_dist",
}

fieldsplit_0_hypre = {
    "ksp_type": "richardson",
    "ksp_max_it": 2,
    "pc_type": "hypre",
}

mg_levels_solver = {
    "ksp_type": "fgmres",
    "ksp_norm_type": "unpreconditioned",
    "ksp_max_it": 5,
    "pc_type": "python",
    "pc_python_type": "matpatch.MatPatch",
}

fieldsplit_0_mg = {
    "ksp_type": "preonly",
    "ksp_norm_type": "unpreconditioned",
    "ksp_convergence_test": "skip",
    "pc_type": "mg",
    "pc_mg_type": "full",
    "mg_levels": mg_levels_solver,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
}

params = {
    "snes_type": "ksponly",
    "snes_monitor": None,
    "mat_type": "nest",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 300,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_1": fieldsplit_1,
}

if args.solver_type == "almg":
    params["fieldsplit_0"] = fieldsplit_0_mg
elif args.solver_type == "allu":
    params["fieldsplit_0"] = fieldsplit_0_lu
elif args.solver_type == "alamg":
    params["fieldsplit_0"] = fieldsplit_0_hypre
elif args.solver_type == "lu":
    params = {
        "snes_type": "ksponly",
        "snes_monitor": None,
        "snes_atol": 1e-6,
        "snes_rtol": 1e-10,
        "mat_type": "aij",
        "pmat_type": "aij",
        "ksp_type": "preonly",
        "ksp_rtol": 1.0e-6,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 300,
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu",
    }
else:
    raise ValueError("please specify almg, allu or alamg for --solver-type")

mu_fun= mu(mh[-1])
appctx = {"nu": mu_fun, "gamma": gamma, "dr":dr, "case":case, "w":w}

# Solve Stoke's equation
def get_transfers():
    V = Z.sub(0)
    Q = Z.sub(1)
    tdim = mesh.topological_dimension()
    vtransfer = PkP0SchoeberlTransfer((mu, gamma), tdim, hierarchy)
    qtransfer = NullTransfer()
    transfers = {V.ufl_element(): (vtransfer.prolong, vtransfer.restrict, inject),
                 Q.ufl_element(): (prolong, restrict, qtransfer.inject)}
    return transfers

nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

if args.nonzero_initial_guess:
    z.split()[0].project(Constant((1., 1.)))
    z.split()[1].interpolate(SpatialCoordinate(mesh)[1]-2)


for i in range(args.itref+1):
    problem = LinearVariationalProblem(a, l, z, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     options_prefix="ns_",
                                     appctx=appctx, nullspace=nsp)

    if args.solver_type == "almg" and args.discretisation == "cg":
        transfermanager = TransferManager(native_transfers=get_transfers())
        solver.set_transfer_manager(transfermanager)
    # Write out solution
    solver.solve()
    with assemble(action(Fgamma, z), bcs=homogenize(bcs)).dat.vec_ro as v:
        PETSc.Sys.Print('Residual with    grad-div', v.norm())
    with assemble(action(F, z), bcs=homogenize(bcs)).dat.vec_ro as w:
        PETSc.Sys.Print('Residual without grad-div', w.norm())

#File("u.pvd").write(z.split()[0])
# uncomment lines below to write out the solution. then run with --case 3 first
# and then with --case 4 after to make sure that the 'manual/triple matrix
# product' augmented lagrangian implementation does the same thing as the
# variational version.

# with DumbCheckpoint(f"u-{args.case}", mode=FILE_UPDATE) as checkpoint:
#     checkpoint.store(z, name="up")
# z3 = z.copy(deepcopy=True)
# with DumbCheckpoint(f"u-{3}", mode=FILE_READ) as checkpoint:
#     checkpoint.load(z3, name="up")
# PETSc.Sys.Print("absolute diff in vel: ", norm(z.split()[0]-z3.split()[0]))
# PETSc.Sys.Print("relative diff in vel: ", norm(z.split()[0]-z3.split()[0])/norm(z.split()[0]))
# PETSc.Sys.Print("absolute diff in pre: ", norm(z.split()[1]-z3.split()[1]))
# PETSc.Sys.Print("relative diff in pre: ", norm(z.split()[1]-z3.split()[1])/norm(z.split()[1]))
#
# File(f"up-{args.case}.pvd").write(*(z.split()))


# if Z.dim() > 1e4 or mesh.mpi_comm().size > 1:
#     import sys; sys.exit()
