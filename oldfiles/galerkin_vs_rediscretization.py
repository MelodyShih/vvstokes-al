from firedrake import *
from alfi.transfer import *
from firedrake.mg.utils import get_level

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
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--nonzero-initial-guess", dest="nonzero_initial_guess", default=False, action="store_true")
parser.add_argument("--discretisation", type=str, default="hdiv")
parser.add_argument("--mattype", type=str, default="aij")
parser.add_argument("--galerkin", dest="galerkin", default=False, action ="store_true")
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
N = args.N
gamma = Constant(args.gamma)

distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

hierarchy = "uniform"
def before(dm, i):
     for p in range(*dm.getHeightStratum(1)):
         dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
     for p in range(*dm.getHeightStratum(1)):
         dm.setLabelValue("prolongation", p, i+2)

def mesh_hierarchy(hierarchy, nref, callbacks, distribution_parameters):
    baseMesh = RectangleMesh(N, N, 4, 4, distribution_parameters=distp)
    if hierarchy == "uniform":
        mh = MeshHierarchy(baseMesh, nref, reorder=True, callbacks=callbacks,
                           distribution_parameters=distribution_parameters)
    else:
        raise NotImplementedError("Only know uniform for the hierarchy.")
    return mh
mh = mesh_hierarchy(hierarchy, nref, (before, after), distp)

mesh = mh[-1]

if args.discretisation == "hdiv":
    V = FunctionSpace(mesh, "BDM", k)
    Q = FunctionSpace(mesh, "DG", k-1)
elif args.discretisation == "cg":
    V = VectorFunctionSpace(mesh, "CG", k)
    Q = FunctionSpace(mesh, "DG", 0)
else:
    raise ValueError("please specify hdiv or cg for --discretisation")

Z = V * Q

PETSc.Sys.Print("dim(V) = ", V.dim())
PETSc.Sys.Print("dim(Q) = ", Q.dim())

sol = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

bcs = [DirichletBC(V, Constant((0., 0.)), "on_boundary")]

omega = 0.1 #0.4, 0.1
delta = 100 #10, 200
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
    return reduce(lambda x, y : x*y, indis, Constant(1.0))

def mu_expr(mesh):
    return (mu_max-mu_min)*(1-chi_n(mesh)) + mu_min

def mu(mesh):
    Qm = FunctionSpace(mesh, Q.ufl_element())
    return Function(Qm).interpolate(mu_expr(mesh))

#File("mu.pvd").write(mu(mesh))

sigma = Constant(100.)
h = CellDiameter(mesh)
n = FacetNormal(mesh)

def diffusion(u, v, mu):
    if args.discretisation == "cg":
        return (mu*inner(2*sym(grad(u)), grad(v)))*dx
    else:
        return (mu*inner(2*sym(grad(u)), grad(v)))*dx \
            - mu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS \
            - mu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS \
            + mu * sigma/avg(h) * inner(2*avg(outer(u,n)),2*avg(outer(v,n))) * dS

def nitsche(u, v, mu, bid, g):
    if args.discretisation == "cg":
        return 0
    else:
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
    Fgamma = F + gamma*inner(div(u), div(v))*dx(degree=2*(k-1))
elif args.discretisation == "cg":
    Fgamma = F + gamma*inner(cell_avg(div(u)), cell_avg(div(v)))*dx(degree=2*(k-1))
else:
    raise ValueError("please specify hdiv or cg for --discretisation")

a = lhs(Fgamma)
l = rhs(Fgamma)


common = {
    "snes_type": "ksponly",
    "mat_type": args.mattype,
    "pmat_type": args.mattype,
    "ksp_type": "fgmres",
    "ksp_gmres_restart ": 300,
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

#mg_levels_solver = {
#    "ksp_type": "fgmres",
#    "ksp_norm_type": "unpreconditioned",
#    "ksp_max_it": 5,
#    "pc_type": "jacobi",
#}

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

if args.nonzero_initial_guess:
    sol.project(Constant((1., 1.)))


def get_prolong():
    if args.discretisation == "cg":
        V = Z.sub(0)
        Q = Z.sub(1)
        tdim = mesh.topological_dimension()
        vtransfer = PkP0SchoeberlTransfer((mu, gamma), tdim, hierarchy)
        return vtransfer.prolong
    else:
        return prolong

def get_transfers():
    V = Z.sub(0)
    Q = Z.sub(1)
    tdim = mesh.topological_dimension()
    vtransfer = PkP0SchoeberlTransfer((mu, gamma), tdim, hierarchy)
    transfers = {V.ufl_element(): (vtransfer.prolong, vtransfer.restrict, inject)}
    return transfers

def build_prolongation_matrix(prolong, V_coarse, V_fine):
    ''' From coarse to fine '''
    uc = Function(V_coarse)
    uf = Function(V_fine)

    ProOp = PETSc.Mat()
    ProOp.create(PETSc.COMM_WORLD)
    ProOp.setSizes([V_fine.dim(), V_coarse.dim()])
    ProOp.setType(args.mattype)
    ProOp.setUp()
    

    for icol in range(V_coarse.dim()):
        uc.dat.zero()
        if args.discretisation == "cg":
            uc.dat.data[int(icol/2)][icol%2] = 1.0
        else:
            uc.dat.data[icol] = 1.0
        arr = uc.vector().get_local()
        prolong(uc, uf)
        values = uf.vector().get_local()
        rows = np.where(np.absolute(values) > 1e-14)[0].astype(np.int32)
        values = values[rows]
        ProOp.setValues(rows, [icol], values)

    ProOp.assemblyBegin()
    ProOp.assemblyEnd()

    ## Save the prolongation matrix
    #viewer = PETSc.Viewer().createBinary("ProOp_"+str(int(V_coarse.dim()))+\
    #                                     "_dr"+str(int(dr))+\
    #                                     "_r"+str(int(args.gamma))+\
    #                                     ".dat",\
    #                                     PETSc.Viewer.Mode.WRITE)
    #viewer.pushFormat(viewer.Format.NATIVE)
    #viewer.view(ProOp)

    return ProOp

# Build level operators
if args.galerkin:
    levelOps = []
    M = assemble(a, bcs=bcs, mat_type=args.mattype)
    Afine = M.petscmat
    levelOps.append(Afine)
    level = nref
    Vf = V
    while level > 0:
        print("building level : ", level)
        Vc = FunctionSpace(mh[level-1], V.ufl_element())
        prolong = get_prolong()
        ProOp = build_prolongation_matrix(prolong, Vc, Vf)
        Acoarse = Afine.PtAP(ProOp)
        #bclevel = DirichletBC(Vc, Constant((0., 0.)), "on_boundary")
        #bc_idx = bclevel.nodes
        #nodes = []
        #for i in bc_idx:
        #    nodes.append(2*i)
        #    nodes.append(2*i+1)
        #Acoarse.zeroRowsColumns(nodes,diag=1)
    
        levelOps.append(Acoarse)
        Afine = Acoarse
        Vf = Vc
        level = level-1

def aug_jacobian(X, J, ctx):
    mh, level = get_level(ctx._x.ufl_domain())
    if args.galerkin:
        rmap, cmap = J.getLGMap()
        levelOps[nref-level].copy(J,structure=J.Structure.DIFFERENT_NONZERO_PATTERN)
        J.setLGMap(rmap, cmap)
        #viewer = PETSc.Viewer().createASCII("LevelOp_"+str(int(level))+\
        #                                     "_galerkin_J.dat",\
        #                                     PETSc.Viewer.Mode.WRITE)
        #viewer.view(J)
    #else:
        #viewer = PETSc.Viewer().createASCII("LevelOp_"+str(int(level))+\
        #                                     ".dat",\
        #                                     PETSc.Viewer.Mode.WRITE)
        #viewer.view(J)

for i in range(args.itref+1):
    problem = LinearVariationalProblem(a, l, sol, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     post_jacobian_callback=aug_jacobian)

    if args.solver_type == "almg" and args.discretisation == "cg":
        transfermanager = TransferManager(native_transfers=get_transfers())
        solver.set_transfer_manager(transfermanager)

    solver.solve()
    with assemble(action(Fgamma, sol), bcs=homogenize(bcs)).dat.vec_ro as v:
        PETSc.Sys.Print('Relative residual with    grad-div', v.norm()/norm(sol))

# Write out solution
#File("u.pvd").write(sol)
