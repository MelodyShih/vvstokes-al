from firedrake import *
from functools import reduce

import argparse
import numpy as np
from petsc4py import PETSc

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--dr", type=float, default=1e8)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--case", type=int, default=3)
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
N = args.N
case = args.case
gamma = Constant(args.gamma)

distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
mesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp)

mh = MeshHierarchy(mesh, nref, reorder=True, distribution_parameters=distp)

mesh = mh[-1]

V = FunctionSpace(mesh, "BDM", k)
Q = FunctionSpace(mesh, "DG", k-1)

sol = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
bcs = [DirichletBC(V, Constant((0., 0.)), "on_boundary")]

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
        cx = 2+np.random.uniform(-1,1)
        cy = 2+np.random.uniform(-1,1)
        indis.append(indi(Constant((cx,cy))))
    indis.append(indi(Constant((cx,cy))))

    return reduce(lambda x, y : x*y, indis, Constant(1.0))

def mu_expr(mesh):
    return (mu_max-mu_min)*(1-chi_n(mesh)) + mu_min

def mu(mesh):
    Qm = FunctionSpace(mesh, Q.ufl_element())
    return Function(Qm).interpolate(mu_expr(mesh))

File("mu.pvd").write(mu(mesh))

sigma = Constant(100.)
h = CellSize(mesh)
n = FacetNormal(mesh)

def diffusion(u, v, mu):
    return (mu*inner(2*sym(grad(u)), grad(v)))*dx \
        - mu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS \
        - mu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS \
        + mu * sigma/avg(h) * inner(2*avg(outer(u,n)),2*avg(outer(v,n))) * dS

def nitsche(u, v, mu, bid, g):
    my_ds = ds if bid == "on_boundary" else ds(bid)
    return -inner(outer(v,n),2*mu*sym(grad(u)))*my_ds \
           -inner(outer(u-g,n),2*mu*sym(grad(v)))*my_ds \
           +mu*(sigma/h)*inner(v,u-g)*my_ds

F = diffusion(u, v, mu_expr(mesh))
for bc in bcs:
    if "DG" in str(bc._function_space):
        continue
    g = bc.function_arg
    bid = bc.sub_domain
    F += nitsche(u, v, mu_expr(mesh), bid, g)

F += -10 * (chi_n(mesh)-1)*v[1] * dx

if case < 4:
    Fgamma = F + gamma*inner(div(u), div(v))*dx
    a = lhs(Fgamma)
    l = rhs(Fgamma)
else:
    raise ValueError("Unknown type of preconditioner %i" % case)

common = {
    "ksp_type": "richardson",
    "ksp_norm_type": "unpreconditioned",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 100,
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
}

solver_lu = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist",
}

solver_hypre = {
    "pc_type": "hypre",
}

mg_levels_solver = {
    "ksp_type": "fgmres",
    "ksp_norm_type": "unpreconditioned",
    "ksp_max_it": 5,
    "pc_type": "python",
    "pc_python_type": "matpatch.MatPatch",
}

solver_mg = {
    "pc_type": "mg",
    "pc_mg_type": "full",
    "mg_levels": mg_levels_solver,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
}

if args.solver_type == "almg":
    params = {**common, **solver_mg}
elif args.solver_type == "allu":
    params = {**common, **solver_lu}
elif args.solver_type == "alamg":
    params = {**common, **solver_hypre}
else:
    raise ValueError("please specify almg, allu or alamg for --solver-type")

mu_fun= mu(mh[-1])
appctx = {"nu": mu_fun, "gamma": gamma, "dr":dr, "case":case}

# Solve Stoke's equation
problem = LinearVariationalProblem(a, l, sol, bcs=bcs)
solver = LinearVariationalSolver(problem,
                                 solver_parameters=params,
                                 options_prefix="topleft_")

# Write out solution
solver.solve()
File("u.pvd").write(sol)


# if Z.dim() > 1e4 or mesh.mpi_comm().size > 1:
#     import sys; sys.exit()
