from firedrake import *
from functools import reduce
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--dr", type=float, default=1e8)
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
gamma = Constant(args.gamma)

distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
mesh = RectangleMesh(10, 10, 4, 4, distribution_parameters=distp)

mh = MeshHierarchy(mesh, nref, reorder=True, distribution_parameters=distp)

mesh = mh[-1]

V = FunctionSpace(mesh, "BDM", k)
Q = FunctionSpace(mesh, "DG", k-1)
Z = V * Q
print("dim(Z) = ", Z.dim())
z = Function(Z)
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
bcs = [DirichletBC(Z.sub(0), Constant((0., 0.)), "on_boundary")]

omega = 0.1
delta = 200
mu_min = Constant(dr**-0.5)
mu_max = Constant(dr**0.5)

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

def chi_n(mesh):
    X = SpatialCoordinate(mesh)
    def indi(ci):
        return 1-exp(-delta * Max(0, sqrt(inner(ci-X, ci-X))-omega/2)**2)
    indis = [indi(Constant((4*(cx+1)/3, 4*(cy+1)/3))) for cx in range(2) for cy in range(2)]
    return reduce(lambda x, y : x*y, indis, Constant(1.0))

def mu_expr(mesh):
    return (mu_max-mu_min)*(1-chi_n(mesh)) + mu_min

def mu(mesh):
    Qm = FunctionSpace(mesh, Q.ufl_element())
    return Function(Qm).interpolate(mu_expr(mesh))

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

F += - p * div(v) * dx - div(u) * q * dx
F += gamma * inner(div(u), div(v))*dx
F += -10 * (chi_n(mesh)-1)*v[1] * dx
a = lhs(F)
l = rhs(F)

fieldsplit_1 = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "schurcomplement.DGMassInv"
}

fieldsplit_0_lu = {
    "ksp_type": "preonly",
    "ksp_max_it": 1,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist",
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

outer = {
    "snes_type": "ksponly",
    "mat_type": "nest",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 100,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_1": fieldsplit_1,
}

if args.solver_type == "almg":
    outer["fieldsplit_0"] = fieldsplit_0_mg
elif args.solver_type == "allu":
    outer["fieldsplit_0"] = fieldsplit_0_lu
elif args.solver_type == "alamg":
    outer["fieldsplit_0"] = fieldsplit_0_hypre
else:
    raise ValueError("please specify almg, allu or alamg for --solver-type")
params = outer
mu_fun= mu(mh[-1])
appctx = {"nu": mu_fun, "gamma": gamma}

nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
problem = LinearVariationalProblem(a, l, z, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="ns_",
                                    appctx=appctx, nullspace=nsp)
solver.solve()
File("u.pvd").write(z.split()[0])


if Z.dim() > 1e4 or mesh.mpi_comm().size > 1:
    import sys; sys.exit()
""" Demo on how to get the assembled """
M = assemble(a, bcs=bcs)
A = M.M[0, 0].handle # A is now a PETSc Mat type
B = M.M[1, 0].handle

Anp = A[:, :] # obtain a dense numpy matrix
Bnp = B[:, :]
