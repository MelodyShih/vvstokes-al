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
parser.add_argument("--nonzero-rhs", dest="nonzero_rhs", default=False, action="store_true")
parser.add_argument("--nonzero-initial-guess", dest="nonzero_initial_guess", default=False, action="store_true")
parser.add_argument("--itref", type=int, default=1)
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
Z = V * Q
print("dim(Z) = ", Z.dim())
print("dim(V) = ", V.dim())
print("dim(Q) = ", Q.dim())
z = Function(Z)
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
bcs = [DirichletBC(Z.sub(0), Constant((0., 0.)), "on_boundary")]

omega = 0.4
delta = 10
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
        # tmp = indi(Constant((cx,cy)))
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

F += - p * div(v) * dx(degree=2*(k-1)) - div(u) * q * dx(degree=2*(k-1))
F += -10 * (chi_n(mesh)-1)*v[1] * dx
if args.nonzero_rhs:
    divrhs = SpatialCoordinate(mesh)[0]-2
else:
    divrhs = Constant(0)
F += divrhs * q * dx(degree=2*(k-1))

Fgamma = F + Constant(gamma)*inner((div(u)-divrhs), div(v))*dx(degree=2*(k-1))

if case < 4:
    ## Case 1,2,3:
    # a = lhs(F)
    # M = assemble(a, bcs=bcs)
    # A = M.M[0, 0].handle
    # B = M.M[1, 0].handle
    # ptrial = TrialFunction(Q)
    # ptest = TestFunction(Q)
    # W  = assemble(Tensor(inner(ptrial, ptest)*dx).inv).M[0,0].handle
    # BTW = B.transposeMatMult(W)
    # BTWB = BTW.matMult(B)
    # BTWB *= args.gamma
    # # Check the correctness of BTWB (should be equal to A2-A)
    # F += gamma*inner(div(u), div(v))*dx
    # M2 = assemble(lhs(F), bcs=bcs)
    # A2 = M2.M[0, 0].handle
    # print((A2 - A - BTWB).norm())
    a = lhs(Fgamma)
    l = rhs(Fgamma)
elif case == 4:
    # Unaugmented system
    a = lhs(F)
    l = rhs(F)

    # Form BTWB
    M = assemble(a, bcs=bcs)
    A = M.M[0, 0].handle
    B = M.M[1, 0].handle
    ptrial = TrialFunction(Q)
    ptest  = TestFunction(Q)
    W = assemble(Tensor(inner(ptrial, ptest)*dx).inv).M[0,0].handle
    BTW = B.transposeMatMult(W)
    BTW *= args.gamma
    BTWB = BTW.matMult(B)
elif case == 5:
    # Unaugmented system
    a = lhs(F)
    l = rhs(F)

    # Form BTWB
    M = assemble(a, bcs=bcs)
    A = M.M[0, 0].handle
    B = M.M[1, 0].handle
    ptrial = TrialFunction(Q)
    ptest  = TestFunction(Q)
    W = assemble(Tensor(1.0/mu(mh[-1])*inner(ptrial, ptest)*dx).inv).M[0,0].handle
    # W = assemble(Tensor(inner(ptrial, ptest)*dx).inv).M[0,0].handle
    BTW = B.transposeMatMult(W)
    BTW *= args.gamma
    BTWB = BTW.matMult(B)
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
        "snes_atol": 1e-10,
        "snes_rtol": 1e-10,
        "mat_type": "aij",
        "pmat_type": "aij",
        "ksp_type": "preonly",
        "ksp_rtol": 1.0e-10,
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
appctx = {"nu": mu_fun, "gamma": gamma, "dr":dr, "case":case}

# Solve Stoke's equation
def aug_jacobian(X, J):
    if case == 4:
        nested_IS = J.getNestISs()
        Jsub = J.getLocalSubMatrix(nested_IS[0][0], nested_IS[0][0])
        Jsub += BTWB
        J.restoreLocalSubMatrix(nested_IS[0][0], nested_IS[0][0], Jsub)
    else:
        return

def modify_residual(X, F):
    if case == 4:
        vel_is = Z._ises[0]
        pre_is = Z._ises[1]
        Fvel = F.getSubVector(vel_is)
        Fpre = F.getSubVector(pre_is)
        Fvel += BTW*Fpre
        F.restoreSubVector(vel_is, Fvel)
        F.restoreSubVector(pre_is, Fpre)
    else:
        return

nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
# nsp = MixedVectorSpaceBasis(Z, [Z[0], VectorSpaceBasis(vecs=[assemble( (1/mu) * TestFunction(Q)]))
if args.nonzero_initial_guess:
    z.split()[0].project(Constant((1., 1.)))
    z.split()[1].interpolate(SpatialCoordinate(mesh)[1]-2)


for i in range(args.itref):
    problem = LinearVariationalProblem(a, l, z, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     options_prefix="ns_",
                                     post_jacobian_callback=aug_jacobian,
                                     post_function_callback=modify_residual,
                                     appctx=appctx, nullspace=nsp)

    # Write out solution
    solver.solve()
    with assemble(action(Fgamma, z), bcs=homogenize(bcs)).dat.vec_ro as v:
        print('Residual with    grad-div', v.norm())
    with assemble(action(F, z), bcs=homogenize(bcs)).dat.vec_ro as w:
        print('Residual without grad-div', w.norm())


# uncomment lines below to write out the solution. then run with --case 3 first
# and then with --case 4 after to make sure that the 'manual/triple matrix
# product' augmented lagrangian implementation does the same thing as the
# variational version.

# with DumbCheckpoint(f"u-{args.case}", mode=FILE_UPDATE) as checkpoint:
#     checkpoint.store(z, name="up")
# z3 = z.copy(deepcopy=True)
# with DumbCheckpoint(f"u-{3}", mode=FILE_READ) as checkpoint:
#     checkpoint.load(z3, name="up")
# print(norm(z.split()[0]-z3.split()[0]))
# print(norm(z.split()[1]-z3.split()[1]))

File(f"up-{args.case}.pvd").write(*(z.split()))


# if Z.dim() > 1e4 or mesh.mpi_comm().size > 1:
#     import sys; sys.exit()

""" Demo on how to get the assembled """
# M = assemble(a, bcs=bcs)
# A = M.M[0, 0].handle # A is now a PETSc Mat type
# B = M.M[1, 0].handle

"""
# Eigenvalue analysis
M = assemble(a, bcs=bcs)
Agamma = M.M[0, 0].handle # A is now a PETSc Mat type
B      = M.M[1, 0].handle
if case == 4:
    Agammanp = Agamma[:,:] + 0.5*(BTWB[:,:] + BTWB.transpose()[:,:])
else:
    Agammanp = Agamma[:, :] # obtain a dense numpy matrix
Bnp      = B[:, :]

## Schur complement of original Sgamma
Sgamma = -np.matmul(np.matmul(Bnp, np.linalg.inv(Agammanp)), Bnp.transpose())

## Schur complement of original S
# Form -BTAinvB
M = assemble(lhs(F), bcs=bcs)
A = M.M[0, 0].handle
Anp = A[:, :] # obtain a dense numpy matrix
S = -np.matmul(np.matmul(Bnp, np.linalg.inv(Anp)), Bnp.transpose())

## Preconditioner of Sgamma
pp = TrialFunction(Q)
qq = TestFunction(Q)

#-M_p(1/nu)^{-1}
viscmass    = assemble(Tensor(-1.0/mu_fun*inner(pp, qq)*dx))
viscmassinv = assemble(Tensor(-1.0/mu_fun*inner(pp, qq)*dx).inv)
viscmass    = viscmass.petscmat
viscmassinv = viscmassinv.petscmat

#-M_p
massinv = assemble(Tensor(-inner(pp, qq)*dx).inv)
massinv = massinv.petscmat

# Pinv
if case == 3:
    Pinv = viscmassinv[:,:] + args.gamma*massinv[:,:]
elif case == 4:
    Pinv = (1.0 + args.gamma)*viscmassinv[:,:]

# Comparison -M_p(1/nu)^{-1}, -M_p^{-1}
MpinvS   = np.matmul(massinv[:,:], S)
eigval, eigvec = np.linalg.eig(MpinvS)
print("-Mp: ")
print("[", np.partition(eigval, 2)[2], ", ", max(eigval), "]")
Amu = max(eigval)
amu = np.partition(eigval, 2)[2]
print("Amu = ", Amu)
print("amu = ", amu)
print((args.gamma + 1)/(args.gamma + 1/amu), (args.gamma + 1)/(args.gamma + 1/Amu))

MpmuinvS = np.matmul(viscmassinv[:,:], S)
eigval, eigvec = np.linalg.eig(MpmuinvS)
print("-Mp(1/mu): ")
print("[", np.partition(eigval, 2)[2], ", ", max(eigval), "]")
Cmu = max(eigval)
cmu = np.partition(eigval, 2)[2]
print("Cmu = ", Cmu)
print("cmu = ", cmu)
print((args.gamma + 1)/(args.gamma + 1/cmu), (args.gamma + 1)/(args.gamma + 1/Cmu))


# MpinvMpmu = np.matmul(massinv[:,:], viscmass[:,:])
eigval, eigvec = np.linalg.eig(MpinvMpmu)
print("MpinvMp(1/mu): ")
print("[", min(eigval), ", ", max(eigval), "]")
print("optimal 1/a:", 1.0/min(eigval))
a = min(eigval)
a = 1/dr**0.5

## Preconditioned system
PinvSgamma = np.matmul(Pinv, Sgamma)
eigval, eigvec = np.linalg.eig(PinvSgamma)

if case == 3:
    print("1/a = ", 1.0/a, "gamma = ", args.gamma)
    dmu = 1 - 1.0/a/args.gamma
    Dmu = 1 + (1 - cmu)/(a*cmu*args.gamma)
    print("Dmu = ", Dmu)
    print("dmu = ", dmu)
elif case == 4:
    dmu = (args.gamma + 1/Cmu)/(args.gamma + 1)
    Dmu = (args.gamma + 1/cmu)/(args.gamma + 1)

print("PinvSgamma: ")
print("[", np.partition(eigval, 2)[2], ", ", max(eigval), "]")
print("1/Dmu = ", 1.0/Dmu)
if abs(dmu) > 1e-15:
    print("1/dmu = ", 1.0/dmu)
"""
