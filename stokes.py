from mpi4py import MPI
comm = MPI.COMM_WORLD
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
PETSc.Log.begin()
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from alfi.transfer import *
from alfi import *
from functools import reduce
from firedrake.mg.utils import get_level
from balance import load_balance, rebalance

import argparse

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
parser.add_argument("--quad-deg", type=int, dest="quad_deg", default=20)
parser.add_argument("--rebalance", dest="rebalance", default=False, action="store_true")
parser.add_argument("--asmbackend", type=str, choices=['tinyasm', 'petscasm'], default="tinyasm")
parser.add_argument("--nsinker", type=int, default=8)
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

distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}


hierarchy = "uniform"
def before(dm, i):
    if i == 0 and args.rebalance:
        rebalance(dm, i) # rebalance the initial coarse mesh
    if dim == 3:
        for p in range(*dm.getHeightStratum(2)):
            dm.setLabelValue("prolongation", p, i+1)
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)
    for p in range(*dm.getDepthStratum(0)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    if args.rebalance:
        rebalance(dm, i) # rebalance all refined meshes
    if dim == 3:
        for p in range(*dm.getHeightStratum(2)):
            dm.setLabelValue("prolongation", p, i+2)
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)
    for p in range(*dm.getDepthStratum(0)):
        dm.setLabelValue("prolongation", p, i+2)

PETSc.Sys.Print("before mesh hierarchy")
def mesh_hierarchy(hierarchy, nref, callbacks, distribution_parameters):
    if dim == 2:
        baseMesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp, \
                quadrilateral=args.quad)
    elif dim == 3:
        if args.quad:
            basemesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp, quadrilateral=True)
            basemh = MeshHierarchy(basemesh, nref, callbacks=callbacks)
            mh = ExtrudedMeshHierarchy(basemh, height=4, base_layer=N)
            return mh
        else:
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
for mesh in mh:
    load_balance(mesh)
PETSc.Sys.Print("after mesh hierarchy")

mesh = mh[-1]
#File('mesh.pvd').write(mesh)
if args.discretisation == "hdiv":
    if args.quad:
        V = FunctionSpace(mesh, "RTCF", k)
        Q = FunctionSpace(mesh, "DQ", k-1)
    else:
        V = FunctionSpace(mesh, "BDM", k)
        Q = FunctionSpace(mesh, "DG", k-1)
elif args.discretisation == "cg":
    if dim == 2:
        if args.quad:
            V = VectorFunctionSpace(mesh, "CG", k)
            Q = FunctionSpace(mesh, "DPC", k-1)
            # Q = FunctionSpace(mesh, "DQ", k-2)
        else:
            V = VectorFunctionSpace(mesh, "CG", k)
            Q = FunctionSpace(mesh, "DG", 0)
    elif dim == 3:
        if args.quad:
            horiz_elt = FiniteElement("CG", quadrilateral, k)
            vert_elt = FiniteElement("CG", interval, k)
            elt = VectorElement(TensorProductElement(horiz_elt, vert_elt))
            V = FunctionSpace(mesh, elt)
            Q = FunctionSpace(mesh, "DPC", k-1)
            # Q = FunctionSpace(mesh, "DQ", k-2)
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
else:
    raise ValueError("please specify hdiv or cg for --discretisation")

PETSc.Sys.Print("Finished FunctionSpaces")
Z = V * Q
size = Z.mesh().mpi_comm().size
PETSc.Sys.Print("dim(Z) = %i (%i per core) " % ( Z.dim(), Z.dim()/size))
PETSc.Sys.Print("dim(V) = %i (%i per core) " % ( V.dim(), V.dim()/size))
PETSc.Sys.Print("dim(Q) = %i (%i per core) " % ( Q.dim(), Q.dim()/size))
PETSc.Sys.Print("Start sparsity")
from sparsity import cache_sparsity
cache_sparsity(Z, V, Q)
PETSc.Sys.Print("Finished sparsity")
z = Function(Z)
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
bcs = [DirichletBC(Z.sub(0), Constant((0.,) * args.dim), "on_boundary")]
if dim == 3 and args.quad:
    bcs += [DirichletBC(Z.sub(0), Constant((0., 0., 0.)), "top"), DirichletBC(Z.sub(0), Constant((0., 0., 0.)), "bottom")]


PETSc.Sys.Print("Created BCs")
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

mu_transfer = mu_expr
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

divdegree = None
F += - p * div(v) * dx(degree=divdegree) - div(u) * q * dx(degree=divdegree)

F += -10 * (chi_n(mesh)-1)*v[1] * dx(degree=deg)
if args.nonzero_rhs:
    divrhs = SpatialCoordinate(mesh)[0]-2
else:
    divrhs = Constant(0)
F += divrhs * q * dx(degree=divdegree)

if args.discretisation == "hdiv":
    Fgamma = F + Constant(gamma)*inner(div(u)-divrhs, div(v))*dx(degree=divdegree)
elif args.discretisation == "cg":
    Fgamma = F + Constant(gamma)*inner(cell_avg(div(u))-divrhs, cell_avg(div(v)))*dx(degree=divdegree, metadata={"mode": "vanilla"})
else:
    raise ValueError("please specify hdiv or cg for --discretisation")

if case == 3:
    a = lhs(Fgamma)
    l = rhs(Fgamma)
else:
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
    "pc_factor_mat_solver_type": "mumps",
}

fieldsplit_0_hypre = {
    "ksp_type": "richardson",
    "ksp_max_it": 2,
    "pc_type": "hypre",
}

mg_levels_solver_rich = {
    "ksp_type": "fgmres",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
}

mg_levels_solver_cheb = {
    "ksp_type": "chebyshev",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
}

mg_levels_solver = {
    # "ksp_monitor_true_residual": None,
    "ksp_type": "fgmres",
    "ksp_norm_type": "unpreconditioned",
    "ksp_max_it": 5,
    "pc_type": "python",
    #"pc_python_type": "hexstar.ASMHexStarPC" if (args.dim == 3 and args.quad == True) else "firedrake.ASMStarPC",
    "pc_python_type": "hexstar.ASMHexStarPC" if (args.dim == 3 and args.quad == True) else "star.ASMStarPlusPC",
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

fieldsplit_0_mg = {
    "ksp_type": "preonly",
    "ksp_norm_type": "unpreconditioned",
    "ksp_convergence_test": "skip",
    "pc_type": "mg",
    "pc_mg_type": "full",
    "pc_mg_log": None,
    #"mg_levels": mg_levels_solver,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
}

params = {
    "snes_type": "ksponly",
    "snes_monitor": None,
    "log_view":  None,
    "mat_type": "nest",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-10,
    "ksp_gmres_restart": 200,
    "ksp_max_it": 200,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_1": fieldsplit_1,
}

if args.solver_type == "almg":
    fieldsplit_0_mg["mg_levels"] = mg_levels_solver
    params["fieldsplit_0"] = fieldsplit_0_mg
elif args.solver_type == "almgcheb":
    fieldsplit_0_mg["mg_levels"] = mg_levels_solver_cheb
    params["fieldsplit_0"] = fieldsplit_0_mg
elif args.solver_type == "almgrich":
    fieldsplit_0_mg["mg_levels"] = mg_levels_solver_rich
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
appctx = {"nu_fun": mu_fun, "dxlist": [dx], "nu_exprlist": [mu_expr(mh[-1])], "gamma": gamma, "dr":dr, "case":case, "w":w, "deg":deg}


BBCTWB_dict = {} # These are of type PETSc.Mat
BBCTW_dict = {} # These are of type PETSc.Mat


if not case == 3:
    for level in range(nref+1):
        levelmesh = mh[level]
        Vlevel = FunctionSpace(levelmesh, V.ufl_element())
        Qlevel = FunctionSpace(levelmesh, Q.ufl_element())
        Zlevel = Vlevel * Qlevel
        tmpp = TrialFunction(Qlevel)
        tmpq = TestFunction(Qlevel)
        if case == 4:
            Wlevel = assemble(Tensor(inner(tmpp, tmpq)*dx).inv, mat_type='aij').petscmat
        elif case == 5:
            Wlevel = assemble(Tensor(1.0/mu_expr(levelmesh)*inner(tmpp, tmpq)*\
                                     dx(degree=deg)).inv, mat_type='aij').petscmat
        elif case == 6:
            Wlevel = w*assemble(Tensor(1.0/mu_expr(levelmesh)*inner(tmpp, tmpq)*dx).inv).petscmat + (1-w)*assemble(Tensor(inner(tmpp, tmpq)*dx).inv).petscmat
        else:
            raise ValueError("Augmented Jacobian (case %d) not implemented yet" % case)

        tmpu, tmpp = TrialFunctions(Zlevel)
        tmpv, tmpq = TestFunctions(Zlevel)
        tmpbcs = [DirichletBC(Zlevel.sub(0), Constant((0.,) * args.dim), "on_boundary")]
        if args.dim == 3 and args.quad:
            tmpbcs += [DirichletBC(Zlevel.sub(0), Constant((0., 0., 0.)), "top"), DirichletBC(Zlevel.sub(0), Constant((0., 0., 0.)), "bottom")]
        BBClevel =  assemble(- tmpq * div(tmpu) * dx(degree=divdegree), bcs=tmpbcs, mat_type='nest').petscmat.getNestSubMatrix(1, 0)
        Wlevel *= gamma
        #todo: fill BBC_dict and W_dict and to the right thin in modify_residual
        if level in BBCTWB_dict:
            BBCTWBlevel = Wlevel.PtAP(BBClevel, result=BBCTWB_dict[level])
        else:
            BBCTWB_dict[level] = Wlevel.PtAP(BBClevel)

PETSc.Sys.Print("Computed BTWB products")


# Solve Stoke's equation
def aug_jacobian(X, J, ctx):
    mh, level = get_level(ctx._x.ufl_domain())
    if case in [4, 5, 6]:
        BTWBlevel = BBCTWB_dict[level]
        if level == nref:
            Jsub = J.getNestSubMatrix(0, 0)
            rmap, cmap = Jsub.getLGMap()
            Jsub.axpy(1, BTWBlevel, Jsub.Structure.SUBSET_NONZERO_PATTERN)
            Jsub.setLGMap(rmap, cmap)
        else:
            rmap, cmap = J.getLGMap()
            J.axpy(1, BTWBlevel, J.Structure.SUBSET_NONZERO_PATTERN)
            J.setLGMap(rmap, cmap)


def modify_residual(X, F):
    if case in [4, 5, 6]:
        vel_is = Z._ises[0]
        pre_is = Z._ises[1]
        Fvel = F.getSubVector(vel_is)
        Fpre = F.getSubVector(pre_is)
        BTW = BBCTW_dict[nref]
        Fvel += BTW*Fpre
        F.restoreSubVector(vel_is, Fvel)
        F.restoreSubVector(pre_is, Fpre)
    else:
        return

def get_transfers(A_callback=None, BTWB_callback=None):
    V = Z.sub(0)
    Q = Z.sub(1)
    tdim = mesh.topological_dimension()
    if case == 3:
        # vtransfer = PkP0SchoeberlTransfer((mu_transfer, gamma), tdim, hierarchy, backend='pcpatch', b_matfree=True, hexmesh=(args.dim == 3 and args.quad))
        vtransfer = PkP0SchoeberlTransfer((mu_transfer, gamma), tdim, hierarchy, backend=args.asmbackend, b_matfree=True, hexmesh=(args.dim == 3 and args.quad))
    else:
        # vtransfer = AlgebraicSchoeberlTransfer((mu_transfer, gamma), A_callback, BTWB_callback, tdim, 'uniform', backend='lu', hexmesh=(args.dim == 3 and args.quad))
        vtransfer = AlgebraicSchoeberlTransfer((mu_transfer, gamma), A_callback, BTWB_callback, tdim, 'uniform', backend=args.asmbackend, hexmesh=(args.dim == 3 and args.quad))
    qtransfer = NullTransfer()
    transfers = {V.ufl_element(): (vtransfer.prolong, vtransfer.restrict, inject),
                 Q.ufl_element(): (prolong, restrict, qtransfer.inject)}
    return transfers

nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

if args.nonzero_initial_guess:
    z.split()[0].project(Constant((1., 1.)))
    z.split()[1].interpolate(SpatialCoordinate(mesh)[1]-2)

PETSc.Sys.Print("Start solves")
for i in range(args.itref+1):
    problem = LinearVariationalProblem(a, l, z, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     options_prefix="ns_",
                                     post_jacobian_callback=aug_jacobian,
                                     post_function_callback=modify_residual,
                                     appctx=appctx, nullspace=nsp)

    if args.solver_type == "almg" and args.discretisation == "cg":
        if case == 3:
            transfers = get_transfers()
        else:
            def BTWBcb(level):
                return BBCTWB_dict[level]
            def Acb(level):
                ctx = solver._ctx
                if level == nref:
                    splitctx = ctx.split([(0,)])[0]
                    A = splitctx._jac
                    A.form = splitctx.J
                    A.M = ctx._jac.M[0, 0]
                    A.petscmat = ctx._jac.petscmat.getNestSubMatrix(0, 0)
                    return A
                else:
                    ksp = solver.snes.ksp.pc.getFieldSplitSubKSP()[0]
                    ctx = get_appctx(ksp.pc.getMGSmoother(level).dm) 
                    A = ctx._jac
                    A.form = ctx.J
                    A.petscmat = A.petscmat
                    return A
            transfers = get_transfers(A_callback=Acb, BTWB_callback=BTWBcb)
        transfermanager = TransferManager(native_transfers=transfers)
        solver.set_transfer_manager(transfermanager)
    # Write out solution
    solver.Z = Z #for calling performance_info
    solver.solve()
    #performance_info(COMM_WORLD, solver)
    #if case==3 or case==4:
    #    with assemble(action(Fgamma, z), bcs=homogenize(bcs)).dat.vec_ro as v:
    #        PETSc.Sys.Print('Residual with    grad-div', v.norm())
    #    with assemble(action(F, z), bcs=homogenize(bcs)).dat.vec_ro as w:
    #        PETSc.Sys.Print('Residual without grad-div', w.norm())

PETSc.Log.view()
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
