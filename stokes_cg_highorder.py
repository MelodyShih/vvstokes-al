from firedrake import *
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
from alfi.transfer import *
from functools import reduce
from firedrake.mg.utils import get_level

import argparse
import numpy as np
from petsc4py import PETSc

import logging
logging.basicConfig(level="INFO")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--case", type=int, default=4)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--dr", type=float, default=1e8)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--nonzero-rhs", dest="nonzero_rhs", default=False, action="store_true")
parser.add_argument("--nonzero-initial-guess", dest="nonzero_initial_guess", default=False, action="store_true")
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--w", type=float, default=0.0)
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
N = args.N
case = args.case
w = args.w
gamma = Constant(args.gamma)

distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}


def before(dm, i):
     for p in range(*dm.getHeightStratum(1)):
         dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
     for p in range(*dm.getHeightStratum(1)):
         dm.setLabelValue("prolongation", p, i+2)

baseMesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp, quadrilateral=True)
mh = MeshHierarchy(baseMesh, nref, callbacks=(before, after), distribution_parameters=distp)

mesh = mh[-1]

V = VectorFunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "DG", k-2)


Z = V * Q
PETSc.Sys.Print("dim(Z) = ", Z.dim())
PETSc.Sys.Print("dim(V) = ", V.dim())
PETSc.Sys.Print("dim(Q) = ", Q.dim())
z = Function(Z)
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
bcs = [DirichletBC(Z.sub(0), Constant((0., 0.)), "on_boundary")]


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

n = FacetNormal(mesh)

def diffusion(u, v, mu):
    return (mu*inner(2*sym(grad(u)), grad(v)))*dx

F = diffusion(u, v, mu_expr(mesh))
F += - p * div(v) * dx - div(u) * q * dx
F += -10 * (chi_n(mesh)-1)*v[1] * dx


if case < 4:
    a = lhs(Fgamma)
    l = rhs(Fgamma)
elif case == 4:
    # Unaugmented system
    a = lhs(F)
    l = rhs(F)

    # Form BTWB
    M = assemble(a, bcs=bcs)
    A = M.M[0, 0].handle
    print("A.block_sizes", A.block_sizes)
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

elif case == 6:
    # Unaugmented system
    a = lhs(F)
    l = rhs(F)

    # Form BTWB
    M = assemble(a, bcs=bcs)
    A = M.M[0, 0].handle
    B = M.M[1, 0].handle
    ptrial = TrialFunction(Q)
    ptest  = TestFunction(Q)
    W1 = assemble(Tensor(1.0/mu(mh[-1])*inner(ptrial, ptest)*dx).inv).M[0,0].handle
    W2 = assemble(Tensor(inner(ptrial, ptest)*dx).inv).M[0,0].handle
    W = W1*w + W2*(1-w)
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
    "ksp_rtol": 1.0e-9,
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
File('/tmp/tmp.pvd').write(Function(Q).interpolate(mu_fun))
appctx = {"nu": mu_fun, "gamma": gamma, "dr":dr, "case":case, "w":w}


def A_callback(level, mat=None):
    levelmesh = mh[level]
    Vlevel = VectorFunctionSpace(levelmesh, "CG", k)
    tmpu = TrialFunction(Vlevel)
    tmpv = TestFunction(Vlevel)
    tmpa = diffusion(tmpu, tmpv, mu_expr(levelmesh))
    tmpbcs = [DirichletBC(Vlevel, Constant((0., 0.)), "on_boundary")]
    M = assemble(tmpa, bcs=tmpbcs, tensor=mat)
    return M


BTWB_dict = {} # These are of type PETSc.Mat
BTW_dict = {} # These are of type PETSc.Mat

for level in range(nref+1):
    levelmesh = mh[level]
    Vlevel = VectorFunctionSpace(levelmesh, "CG", k)
    Qlevel = FunctionSpace(levelmesh, "DG", k-2)
    Zlevel = Vlevel * Qlevel
    tmpp = TrialFunction(Qlevel)
    tmpq = TestFunction(Qlevel)
    Wlevel = assemble(Tensor(tmpp * tmpq * dx).inv).petscmat
    tmpu, tmpp = TrialFunctions(Zlevel)
    tmpv, tmpq = TestFunctions(Zlevel)
    tmpbcs = [DirichletBC(Zlevel.sub(0), Constant((0., 0.)), "on_boundary")]
    Blevel =  assemble(- tmpq * div(tmpu) * dx, bcs=tmpbcs).M[1, 0].handle
    if level in BTW_dict:
        BTWlevel = Blevel.transposeMatMult(Wlevel, result=BTW_dict[level])
    else:
        BTWlevel = Blevel.transposeMatMult(Wlevel)
        BTW_dict[level] = BTWlevel
    BTWlevel *= args.gamma
    if level in BTWB_dict:
        BTWBlevel = BTWlevel.matMult(Blevel, result=BTWB_dict[level])
    else:
        BTWBlevel = BTWlevel.matMult(Blevel)
        BTWB_dict[level] = BTWBlevel

def BTWB_callback(level, mat=None):
    return BTWB_dict[level]

# Solve Stoke's equation
def aug_jacobian(X, J, ctx):
    mh, level = get_level(ctx._x.ufl_domain())
    if case == 4 or case == 5:
        BTWBlevel = BTWB_dict[level]
        if level == nref:
            Jsub = J.getNestSubMatrix(0, 0)
            rmap, cmap = Jsub.getLGMap()
            Jsub.axpy(1, BTWBlevel, Jsub.Structure.SUBSET_NONZERO_PATTERN)
            Jsub.setLGMap(rmap, cmap)
        else:
            rmap, cmap = J.getLGMap()
            J.axpy(1, BTWBlevel, J.Structure.SUBSET_NONZERO_PATTERN)
            J.setLGMap(rmap, cmap)
    elif case == 6:
        raise ValueError("Augmented Jacobian (case %d) not implemented yet" % case)


def modify_residual(X, F):
    if case == 4 or case == 5 or case == 6:
        vel_is = Z._ises[0]
        pre_is = Z._ises[1]
        Fvel = F.getSubVector(vel_is)
        Fpre = F.getSubVector(pre_is)
        BTW = BTW_dict[nref]
        Fvel += BTW*Fpre
        F.restoreSubVector(vel_is, Fvel)
        F.restoreSubVector(pre_is, Fpre)
    else:
        return

def get_transfers():
    V = Z.sub(0)
    Q = Z.sub(1)
    tdim = mesh.topological_dimension()
    # vtransfer = PkP0SchoeberlTransfer((mu, gamma), tdim, 'uniform')
    vtransfer = AlgebraicSchoeberlTransfer((mu, gamma), A_callback, BTWB_callback, tdim, 'uniform')
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
                                     post_jacobian_callback=aug_jacobian,
                                     post_function_callback=modify_residual,
                                     appctx=appctx, nullspace=nsp)

    if args.solver_type == "almg":
        transfermanager = TransferManager(native_transfers=get_transfers())
        solver.set_transfer_manager(transfermanager)
    # Write out solution
    solver.solve()
    if case==3 or case==4:
        with assemble(action(F, z), bcs=homogenize(bcs)).dat.vec_ro as w:
            PETSc.Sys.Print('Residual without grad-div', w.norm())
        # with assemble(action(Fgamma, z), bcs=homogenize(bcs)).dat.vec_ro as v:
        #     PETSc.Sys.Print('Residual with    grad-div', v.norm())

File("u.pvd").write(z.split()[0])
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
