from firedrake import *
from firedrake.petsc import PETSc
from alfi.transfer import *
from alfi import *
from functools import reduce
from firedrake.mg.utils import get_level

import argparse
import numpy as np
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()

import logging
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
parser.add_argument("--quad", dest="quad", default=False, action="store_true")
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--w", type=float, default=0.0)
parser.add_argument("--discretisation", type=str, default="hdiv")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--quad-deg", type=int, dest="quad_deg", default=8)
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
N = args.N
case = args.case
w = args.w
gamma = Constant(args.gamma)
dim = args.dim
deg = args.quad_deg

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
        baseMesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp, \
                quadrilateral=args.quad)
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

if args.quad:
    V = FunctionSpace(mesh, "RTCF", k)
    Q = FunctionSpace(mesh, "DQ", k-1)
else:
    if args.discretisation == "hdiv":
        V = FunctionSpace(mesh, "BDM", k)
        Q = FunctionSpace(mesh, "DG", k-1)
    elif args.discretisation == "cg":
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

sigma = Constant(100.)
if args.quad:
    h = CellDiameter(mesh)
else:
    if args.discretisation == "hdiv":
        #h = CellDiameter(mesh)
        h = Constant(sqrt(2)/(N*(2**nref)))
    elif args.discretisation == "cg":
        h = CellDiameter(mesh)
    else:
        raise ValueError("please specify hdiv or cg for --discretisation")
n = FacetNormal(mesh)

def diffusion(u, v, mu):
    if args.discretisation == "cg":
        return (mu*inner(2*sym(grad(u)), grad(v)))*dx(degree=deg)
    else:
        return (mu*inner(2*sym(grad(u)), grad(v)))*dx(degree=deg)\
            - mu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS(degree=deg) \
            - mu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS(degree=deg) \
            + mu * sigma/avg(h) * inner(2*avg(outer(u,n)),2*avg(outer(v,n))) * dS(degree=deg)

def nitsche(u, v, mu, bid, g):
    if args.discretisation == "cg":
        return 0
    else:
        my_ds = ds if bid == "on_boundary" else ds(bid)
        return -inner(outer(v,n),2*mu*sym(grad(u)))*my_ds(degree=deg) \
               -inner(outer(u-g,n),2*mu*sym(grad(v)))*my_ds(degree=deg) \
               +mu*(sigma/h)*inner(v,u-g)*my_ds(degree=deg)

F = diffusion(u, v, mu_expr(mesh))
for bc in bcs:
    if "DG" in str(bc._function_space):
        continue
    g = bc.function_arg
    bid = bc.sub_domain
    F += nitsche(u, v, mu_expr(mesh), bid, g)

if dim == 2:
    F += - p * div(v) * dx(degree=2*(k-1)) - div(u) * q * dx(degree=2*(k-1))
elif dim == 3:
    F += - p * div(v) * dx(degree=3*(k-1)) - div(u) * q * dx(degree=3*(k-1))

F += -10 * (chi_n(mesh)-1)*v[1] * dx(degree=deg)
if args.nonzero_rhs:
    divrhs = SpatialCoordinate(mesh)[0]-2
else:
    divrhs = Constant(0)
F += divrhs * q * dx(degree=2*(k-1))

if args.quad:
    Fgamma = F + Constant(gamma)*inner(div(u)-divrhs, div(v))*dx(degree=2*(k-1))
else:
    if args.discretisation == "hdiv":
        Fgamma = F + Constant(gamma)*inner(div(u)-divrhs, div(v))*dx(degree=3*(k-1))
    elif args.discretisation == "cg":
        Fgamma = F + Constant(gamma)*inner(cell_avg(div(u))-divrhs, cell_avg(div(v)))*dx(degree=3*(k-1))
    else:
        raise ValueError("please specify hdiv or cg for --discretisation")

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
    W = assemble(Tensor(1.0/mu(mh[-1])*inner(ptrial, ptest)*dx(degree=deg)).inv).M[0,0].handle
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
appctx = {"nu": mu_fun, "gamma": gamma, "dr":dr, "case":case, "w":w, "deg":deg}

# Solve Stoke's equation
def aug_jacobian(X, J, ctx):
    mh, level = get_level(ctx._x.ufl_domain())
    levelmesh = mh[level]
    if case == 4 or case == 5:
        levelmesh = mh[level]
        if args.quad:
            Vlevel = FunctionSpace(levelmesh, "RTCF", k)
            Qlevel = FunctionSpace(levelmesh, "DQ", k-1)
        else:
            if args.discretisation == "hdiv":
                Vlevel = FunctionSpace(levelmesh, "BDM", k)
                Qlevel = FunctionSpace(levelmesh, "DG", k-1)
            elif args.discretisation == "cg":
                if dim == 2:
                    Vlevel = VectorFunctionSpace(levelmesh, "CG", k)
                    Qlevel = FunctionSpace(levelmesh, "DG", 0)
                elif dim == 3:
                    Pklevel = FiniteElement("Lagrange", levelmesh.ufl_cell(), 2)
                    FBlevel = FiniteElement("FacetBubble", levelmesh.ufl_cell(), 3)
                    eleulevel = VectorElement(NodalEnrichedElement(Pklevel, FBlevel))
                    Vlevel = FunctionSpace(levelmesh, eleulevel)
                    Qlevel = FunctionSpace(levelmesh, "DG", 0)
                else:
                    raise NotImplementedError("Only implemented for dim=2,3")
            else:
                raise ValueError("please specify hdiv or cg for --discretisation")
        Zlevel = Vlevel * Qlevel
        # Get B
        tmpu, tmpp = TrialFunctions(Zlevel)
        tmpv, tmpq = TestFunctions(Zlevel)
        tmpF = -tmpq * div(tmpu) * dx
        if dim == 2:
            tmpbcs = [DirichletBC(Zlevel.sub(0), Constant((0., 0.)), "on_boundary")]
        elif dim == 3:
            tmpbcs = [DirichletBC(Zlevel.sub(0), Constant((0., 0., 0.)), "on_boundary")]
        else:
            raise NotImplementedError("Only implemented for dim=2,3")
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
                    dx(degree=deg)).inv).M[0,0].handle

        # Form BTWB
        BTWlevel = Blevel.transposeMatMult(Wlevel)
        BTWlevel *= args.gamma
        BTWBlevel = BTWlevel.matMult(Blevel)

        if level == nref:
            vel_is = Z._ises[0]
            Jsub = J.getLocalSubMatrix(vel_is, vel_is)
            if args.discretisation == "hdiv":
                Jsub.axpy(1, BTWBlevel, structure=Jsub.Structure.SUBSET_NONZERO_PATTERN)
            elif args.discretisation == "cg":
                Jsub.axpy(1, BTWBlevel)
            J.restoreLocalSubMatrix(vel_is, vel_is, Jsub)
        else:
            if args.discretisation == "hdiv":
                J.axpy(1, BTWBlevel, structure=J.Structure.SUBSET_NONZERO_PATTERN)
            elif args.discretisation == "cg":
                J.axpy(1, BTWBlevel)
    elif case == 6:
        raise ValueError("Augmented Jacobian (case %d) not implemented yet" % case)


def modify_residual(X, F):
    if case == 4 or case == 5 or case == 6:
        vel_is = Z._ises[0]
        pre_is = Z._ises[1]
        Fvel = F.getSubVector(vel_is)
        Fpre = F.getSubVector(pre_is)
        Fvel += BTW*Fpre
        F.restoreSubVector(vel_is, Fvel)
        F.restoreSubVector(pre_is, Fpre)
    else:
        return

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
    PETSc.Log.begin()
    problem = LinearVariationalProblem(a, l, z, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     options_prefix="ns_",
                                     post_jacobian_callback=aug_jacobian,
                                     post_function_callback=modify_residual,
                                     appctx=appctx, nullspace=nsp)

    if not args.quad:
        if args.solver_type == "almg" and args.discretisation == "cg":
            transfermanager = TransferManager(native_transfers=get_transfers())
            solver.set_transfer_manager(transfermanager)
    # Write out solution
    solver.Z = Z #for calling performance_info
    solver.solve()
    performance_info(COMM_WORLD, solver)
    #if case==3 or case==4:
    #    with assemble(action(Fgamma, z), bcs=homogenize(bcs)).dat.vec_ro as v:
    #        PETSc.Sys.Print('Residual with    grad-div', v.norm())
    #    with assemble(action(F, z), bcs=homogenize(bcs)).dat.vec_ro as w:
    #        PETSc.Sys.Print('Residual without grad-div', w.norm())

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
