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
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--nonzero-initial-guess", dest="nonzero_initial_guess", default=False, action="store_true")
parser.add_argument("--discretisation", type=str, default="hdiv")
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

if args.discretisation == "hdiv":
    V = FunctionSpace(mesh, "BDM", k)
    Q = FunctionSpace(mesh, "DG", k-1)
elif args.discretisation == "cg":
    assert k == 2, "only k=2 is implemented"
    V = VectorFunctionSpace(mesh, "CG", k)
    Q = FunctionSpace(mesh, "DG", k-2)
else:
    raise ValueError("please specify hdiv or cg for --discretisation")
    
Z = V * Q

print("dim(V) = ", V.dim())
print("dim(Q) = ", Q.dim())

sol = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
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

#File("mu.pvd").write(mu(mesh))

sigma = Constant(100.)
h = CellSize(mesh)
#h = Constant(sqrt(2)/(N*(2**nref)))
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
if args.discretisation == "hdiv":
    Fgamma = F + gamma*inner(div(u), div(v))*dx
elif args.discretisation == "cg":
    assert k == 2, "only k=2 is implemented"
    Fgamma = F + gamma*inner(cell_avg(div(u)), div(v))*dx
else:
    raise ValueError("please specify hdiv or cg for --discretisation")

if case < 4:
    a = lhs(Fgamma)
    l = rhs(Fgamma)
elif case == 4 or case == 5:
    # Unaugmented system
    a = lhs(F)
    l = rhs(F)

    # Get B
    tmpu, tmpp = TrialFunctions(Z)
    tmpv, tmpq = TestFunctions(Z)
    tmpF = -tmpq * div(tmpu) * dx
    tmpbcs = [DirichletBC(Z.sub(0), Constant((0., 0.)), "on_boundary")]
    tmpa = lhs(tmpF)
    M = assemble(tmpa, bcs=tmpbcs)
    B = M.M[1, 0].handle

    # Get W
    ptrial = TrialFunction(Q)
    ptest  = TestFunction(Q)
    if case == 4:
        W = assemble(Tensor(inner(ptrial, ptest)*dx).inv).M[0,0].handle
    if case == 5:
        W = assemble(Tensor(1.0/mu(mh[-1])*inner(ptrial, ptest)*dx).inv).M[0,0].handle

    # Form BTWB
    BTW = B.transposeMatMult(W)
    BTW *= args.gamma
    BTWB = BTW.matMult(B)
else:
    raise ValueError("Unknown type of preconditioner %i" % case)

common = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
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
    #"pc_mg_type": "multiplicative",
    #"pc_mg_cycle_type": "v",
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

def aug_jacobian(X, J, level):
    if case == 4 or case == 5:
        levelmesh = mh[level]
        if args.discretisation == "hdiv":
            Vlevel = FunctionSpace(mesh, "BDM", k)
            Qlevel = FunctionSpace(mesh, "DG", k-1)
        elif args.discretisation == "cg":
            assert k == 2, "only k=2 is implemented"
            Vlevel = VectorFunctionSpace(mesh, "CG", k)
            Qlevel = FunctionSpace(mesh, "DG", k-2)
        else:
            raise ValueError("please specify hdiv or cg for --discretisation")
        Zlevel = Vlevel * Qlevel
        # Get B
        tmpu, tmpp = TrialFunctions(Zlevel)
        tmpv, tmpq = TestFunctions(Zlevel)
        tmpF = -tmpq * div(tmpu) * dx
        tmpbcs = [DirichletBC(Zlevel.sub(0), Constant((0., 0.)), "on_boundary")]
        tmpa = lhs(tmpF)
        M = assemble(tmpa, bcs=tmpbcs)
        Blevel = M.M[1, 0].handle

        # Get W
        ptrial = TrialFunction(Qlevel)
        ptest  = TestFunction(Qlevel)
        if case == 4:
            Wlevel = assemble(Tensor(inner(ptrial, ptest)*dx).inv).M[0,0].handle
        if case == 5:
            Wlevel = assemble(Tensor(1.0/mu(levelmesh)*inner(ptrial, ptest)*\
                    dx).inv).M[0,0].handle

        # Form BTWB
        BTWlevel = Blevel.transposeMatMult(Wlevel)
        BTWlevel *= args.gamma
        BTWBlevel = BTWlevel.matMult(Blevel)
        J.axpy(1, BTWBlevel, structure=J.Structure.SUBSET_NONZERO_PATTERN)

def modify_residual(X, F):
    if case == 4 or case == 5:
        vel_is = Z._ises[0]
        pre_is = Z._ises[1]
        Fvel = F.getSubVector(vel_is)
        Xvel = X.getSubVector(vel_is)
        Fvel += BTWB*Xvel
        F.restoreSubVector(vel_is, Fvel)
    else:
        return

if args.nonzero_initial_guess:
    sol.project(Constant((1., 1.)))

from firedrake.dmhooks import get_appctx
def form(V):
    a = get_appctx(V.dm).J
    return a

def energy_norm(u):
    return assemble(action(action(form(u.function_space()), u), u))

from enum import IntEnum
class Op(IntEnum):
    PROLONG = 0
    RESTRICT = 1
    INJECT = 2

class MyTransferManager(TransferManager):
    def prolong(self, uc, uf):
        """Prolong a function.

        :arg uc: The source (coarse grid) function.
        :arg uf: The target (fine grid) function.
        """
        print("From mesh %i to %i" % (uc.function_space().dim(), uf.function_space().dim()))
        #print("energy_norm(u_H)   ", energy_norm(uc))
        super().prolong(uc,uf)
        #print("energy_norm(P_hu_H)", energy_norm(uf))
        #print("energy ratio:")
        print("   energy ratio:  ", energy_norm(uf)/energy_norm(uc))
        #print()

    def inject(self, uf, uc):
        """Inject a function (primal restriction)

        :arg uf: The source (fine grid) function.
        :arg uc: The target (coarse grid) function.
        """
        if get_appctx(uc.function_space().dm) is not None:
            print("From mesh %i to %i" % (uf.function_space().dim(), uc.function_space().dim()))
            print("energy_norm(u_h)   ", energy_norm(uf))
            super().inject(uf,uc)
            print("energy_norm(I_Hu_h)", energy_norm(uc))
            print("energy ratio:")
            if abs(energy_norm(uf))<1e-15:
                print("   nan")
            else:
                print("  ", energy_norm(uc)/energy_norm(uf))
            print()
        else:
            super().inject(uf,uc)

for i in range(args.itref+1):
    transfer = MyTransferManager()
    problem = LinearVariationalProblem(a, l, sol, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     options_prefix="topleft_",
                                     post_jacobian_callback=aug_jacobian, 
                                     post_function_callback=modify_residual)
    solver.set_transfer_manager(transfer)
    solver.solve()
    if case <= 4:
        with assemble(action(Fgamma, sol), bcs=homogenize(bcs)).dat.vec_ro as v:
            print('Relative residual with    grad-div', v.norm()/norm(sol))

    # Write out solution
File("u.pvd").write(sol)
# if Z.dim() > 1e4 or mesh.mpi_comm().size > 1:
#     import sys; sys.exit()
