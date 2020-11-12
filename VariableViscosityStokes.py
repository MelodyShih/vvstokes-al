'''
=======================================
Implementation of class VariableViscosityStokesProblem, VariableViscosityStokesSolver

Author:                Florian Wechsung
                       Melody Shih
=======================================
'''

from firedrake import *
from firedrake.petsc import PETSc
from alfi.transfer import *
from alfi import *
from firedrake.mg.utils import get_level
from balance import load_balance, rebalance

import numpy as np

class VariableViscosityStokesProblem():
    def create_basemesh(self,basemeshtype,Nx=-1, Ny=-1, Nz=-1, 
                        Lx=-1, Ly=-1, Lz=-1, Cr=-1):
        distp = {"partition": True, 
                 "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
        dim = self.dim
        quad = self.quad
        if 'rectangle' == basemeshtype:
            self.Nx = Nx
            self.Ny = Ny
            self.Nz = Nz
            self.Lx = Lx
            self.Ly = Ly
            self.Lz = Lz
            if dim == 2:
                baseMesh = RectangleMesh(Nx, Ny, Lx, Ly, 
                               distribution_parameters=distp,
                               quadrilateral=quad)
            elif dim == 3:
                if self.quad:
                    baseMesh = RectangleMesh(Nx, Ny, Lx, Ly, 
                               distribution_parameters=distp, 
                               quadrilateral=True)
                else:
                    baseMesh = BoxMesh(Nx, Ny, Nz, Lx, Ly, Lz, 
                                       distribution_parameters=distp)
            else:
                raise NotImplementedError("Only implemented for dim=2,3")
        elif 'rectanglewithhole' == basemeshtype:
            if dim == 2:
                baseMesh = RectangleMesh(Nx, Ny, Lx, Ly, 
                                         distribution_parameters=distp,
                                         quadrilateral=quad)
            else:
                raise NotImplementedError("Only implemented for dim=2")
        else:
            raise NotImplementedError('Unknown type of mesh "%s"'%basemeshtype)
        return baseMesh

    def set_meshhierarchy(self, basemesh, nref, rebalance=False):
        dim = self.dim
        quad = self.quad
        distp = {"partition": True, 
                 "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
        
        def before(dm, i):
            if i == 0 and rebalance:
                rebalance(dm, i) # rebalance the initial coarse mesh
            if dim == 3:
                for p in range(*dm.getHeightStratum(2)):
                    dm.setLabelValue("prolongation", p, i+1)
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+1)
            for p in range(*dm.getDepthStratum(0)):
                dm.setLabelValue("prolongation", p, i+1)

        def after(dm, i):
            if rebalance:
                rebalance(dm, i) # rebalance all refined meshes
            if dim == 3:
                for p in range(*dm.getHeightStratum(2)):
                    dm.setLabelValue("prolongation", p, i+2)
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+2)
            for p in range(*dm.getDepthStratum(0)):
                dm.setLabelValue("prolongation", p, i+2)

        if dim == 3:
            if self.quad:
                basemh = MeshHierarchy(basemesh, nref, callbacks=(before,after))
                mh = ExtrudedMeshHierarchy(basemh, height=self.Lz, 
                                           base_layer=self.Nz)
        else:
            mh = MeshHierarchy(basemesh, nref, reorder=True, 
                               callbacks=(before,after),
                               distribution_parameters=distp)
        for mesh in mh:
            load_balance(mesh)
        self.nref = nref
        self.mh = mh
        self.mesh = mh[-1]

    def get_mesh(self):
        return self.mesh

    def get_meshhierarchy(self):
        return self.mh
        
    def get_functionspace(self, mesh, info=False, dualFncSp=False):
        k = self.k
        quad = self.quad
        dim = self.dim
        discretisation = self.discretisation
        if discretisation == "hdiv":
            if quad:
                V  = FunctionSpace(mesh, "RTCF", k)
                Q  = FunctionSpace(mesh, "DQ", k-1)
                Vd_e = TensorElement('DQ', mesh.ufl_cell(), k-1)
                Vd = FunctionSpace(mesh, Vd_e)
            else:
                V  = FunctionSpace(mesh, "BDM", k)
                Q  = FunctionSpace(mesh, "DG", k-1)
                Vd_e = TensorElement('DG', mesh.ufl_cell(), k-1)
                Vd = FunctionSpace(mesh, Vd_e)
        elif discretisation == "cg":
            if dim == 2:
                if quad:
                    V  = VectorFunctionSpace(mesh, "CG", k)
                    Vd_e = TensorElement('DQ', mesh.ufl_cell(), k-1)
                    Vd = FunctionSpace(mesh, Vd_e)
                    Q  = FunctionSpace(mesh, "DPC", k-1)
                    # Q = FunctionSpace(mesh, "DQ", k-2)
                else:
                    V  = VectorFunctionSpace(mesh, "CG", k)
                    Vd_e = TensorElement('DG', mesh.ufl_cell(), k-1)
                    Vd = FunctionSpace(mesh, Vd_e)
                    Q  = FunctionSpace(mesh, "DG", 0)
            elif dim == 3:
                if quad:
                    horiz_elt = FiniteElement("CG", quadrilateral, k)
                    vert_elt = FiniteElement("CG", interval, k)
                    elt = VectorElement(TensorProductElement(horiz_elt, vert_elt))
                    V = FunctionSpace(mesh, elt)
                    Q = FunctionSpace(mesh, "DPC", k-1)
                    Vd = None
                    # Q = FunctionSpace(mesh, "DQ", k-2)
                else:
                    Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
                    if k < 3:
                        FB = FiniteElement("FacetBubble", mesh.ufl_cell(), 3)
                        eleu = VectorElement(NodalEnrichedElement(Pk, FB))
                    else:
                        eleu = VectorElement(Pk)
                    V = FunctionSpace(mesh, eleu)
                    Vd_e = TensorElement('DG', mesh.ufl_cell(), k-1)
                    Vd = FunctionSpace(mesh, Vd_e)
                    Q = FunctionSpace(mesh, "DG", 0)
            else:
                raise NotImplementedError("Only implemented for dim=2,3")
        else:
            raise ValueError("please specify hdiv or cg for --discretisation")
        if info:
            Z = V*Q
            size = Z.mesh().mpi_comm().size
            PETSc.Sys.Print("dim(Z) = %i (%i per core) " \
                                             % ( Z.dim(), Z.dim()/size))
            PETSc.Sys.Print("dim(V) = %i (%i per core) " \
                                             % ( V.dim(), V.dim()/size))
            PETSc.Sys.Print("dim(Q) = %i (%i per core) " \
                                             % ( Q.dim(), Q.dim()/size))
        if dualFncSp is True:
            return V, Q, Vd
        else: 
            return V, Q

    def create_dirichletbcsfun(self, mesh):
        def dirichletbc_fun(mesh):
            dim = self.dim 
            quad = self.quad
            V,Q = self.get_functionspace(mesh)
            Z = V*Q
            bcs = [DirichletBC(Z.sub(0), Constant((0.,) * dim), "on_boundary")]
            if dim == 3 and quad:
                bcs += [DirichletBC(Z.sub(0), Constant((0., 0., 0.)), "top"),
                        DirichletBC(Z.sub(0), Constant((0., 0., 0.)), "bottom")]
            return bcs
        return dirichletbc_fun

    def set_bcsfun(self, bc_fun):
        self.bc_fun = bc_fun 

    def get_bcs(self, mesh):
        return self.bc_fun(mesh) 

    def get_weakform_augterm(self, mesh, gamma, divrhs):
        divdegree = self.quaddivdeg
        discretisation = self.discretisation
        V,Q = self.get_functionspace(mesh)
        u = TrialFunction(V)
        v = TestFunction(V)
        if discretisation == "hdiv":
            aug = Constant(gamma)*inner(div(u)-divrhs, div(v))*dx(degree=divdegree)
        elif discretisation == "cg":
            aug = Constant(gamma)*inner(cell_avg(div(u))-divrhs, \
                                        cell_avg(div(v)))*dx(degree=divdegree, \
                                                  metadata={"mode": "vanilla"})
        else:
            raise ValueError("please specify hdiv or cg for --discretisation")
        return aug

    def get_weakform_stokes(self, mesh, bcs=None):
        deg = self.quaddeg
        divdegree = None
        V, Q = self.get_functionspace(mesh)
        Z = V*Q
        u,p = TrialFunctions(Z)
        v,q = TestFunctions(Z)
        mu = self.mu_fun(mesh)

        if self.discretisation == "cg":
            # (1,1) block
            F = (mu*inner(2*sym(grad(u)), grad(v)))*dx(degree=deg)
            # (1,2), (2,1) block
            F += -p*div(v)*dx(degree=divdegree)-div(u)*q*dx(degree=divdegree)
        elif self.discretisation == "hdiv":
            sigma = Constant(100.)
            n = FacetNormal(mesh)
            if self.quad:
                h = CellDiameter(mesh)
            else:
                h = Constant(sqrt(2)/(N*(2**nref)))
            def nitsche(u, v, mu, bid, g):
                my_ds = ds if bid == "on_boundary" else ds(bid)
                return -inner(outer(v,n),2*mu*sym(grad(u)))*my_ds(degree=deg) \
                       -inner(outer(u-g,n),2*mu*sym(grad(v)))*my_ds(degree=deg) \
                       +mu*(sigma/h)*inner(v,u-g)*my_ds(degree=deg)
            # (1,1) block with stablization
            F = (mu*inner(2*sym(grad(u)), grad(v)))*dx(degree=deg)\
              - mu*inner(avg(2*sym(grad(u))),2*avg(outer(v, n)))*dS(degree=deg)\
              - mu*inner(avg(2*sym(grad(v))),2*avg(outer(u, n)))*dS(degree=deg)\
              + mu*sigma/avg(h)*inner(2*avg(outer(u,n)),\
                                      2*avg(outer(v,n)))*dS(degree=deg)
            # (1,2), (2,1) block
            F += -p*div(v)*dx(degree=divdegree)-div(u)*q*dx(degree=divdegree)
            # Stablization
            for bc in bcs:
                if "DG" in str(bc._function_space):
                    continue
                g = bc.function_arg
                bid = bc.sub_domain
                F += nitsche(u, v, mu, bid, g)
        else:
            raise ValueError("unknown discretisation %s" %self.discretisation)
        return F
            
    def set_viscosity(self, mu_fun, mu_max, mu_min): 
        self.mu_fun = mu_fun
        self.mu_max = mu_max
        self.mu_min = mu_min

    def get_viscosity(self): 
        return self.mu_fun

    def get_A_weak(self, u, v, mu):
        deg = self.quaddeg
        discretisation = self.discretisation
        if discretisation == "cg":
            return (mu*inner(2*sym(grad(u)), grad(v)))*dx(degree=deg)
        else:
            return (mu*inner(2*sym(grad(u)), grad(v)))*dx(degree=deg)\
                - mu*inner(avg(2*sym(grad(u))),2*avg(outer(v, n)))\
                                                          *dS(degree=deg) \
                - mu*inner(avg(2*sym(grad(v))),2*avg(outer(u, n)))\
                                                          *dS(degree=deg) \
                + mu*sigma/avg(h)*inner(2*avg(outer(u,n)),\
                                        2*avg(outer(v,n)))*dS(degree=deg)

    def get_W_mat(self, mesh, case, w):
        deg = self.quaddeg
        mu = self.mu_fun(mesh)
        V, Q = self.get_functionspace(mesh)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        if case == 4:
            W = assemble(Tensor(inner(p,q)*dx).inv, mat_type='aij').petscmat
        elif case == 5:
            W = assemble(Tensor(1.0/mu*inner(p,q)*dx(degree=deg)).inv,\
                                        mat_type='aij').petscmat
        elif case == 6:
            W = w*assemble(Tensor(1.0/mu*inner(p,q)*dx).inv).petscmat +\
                (1-w)*assemble(Tensor(inner(p,q)*dx).inv).petscmat
        else:
            raise ValueError("Augmented Jacobian (case %d) not implemented yet"\
                                                                        % case)
        return W

    def get_B_mat(self, mesh):
        divdegree = self.quaddivdeg
        V, Q = self.get_functionspace(mesh)
        Z = V*Q
        u, p = TrialFunctions(Z)
        v, q = TestFunctions(Z)
        bcs = self.get_bcs(mesh)
        BBC = assemble(-q * div(u) * dx(degree=divdegree),\
                   bcs=bcs,mat_type='nest').petscmat.getNestSubMatrix(1, 0)
        return BBC

    def set_linearvariationalproblem(self, a, l, z, bcs):
        self.lvproblem = LinearVariationalProblem(a, l, z, bcs=bcs)
        

    def __init__(self, dim, quad, discretisation, discdegree, quaddegree=20,
                 quaddivdegree=None):
        self.dim=dim
        self.discretisation=discretisation
        self.k=discdegree
        self.quad=quad
        self.quaddeg = quaddegree
        self.quaddivdeg = quaddivdegree
        self.mu_fun = None
        self.mu_min = -1
        self.mu_max = -1
        self.nref = -1
        self.baseN = -1
        self.mh = None
        self.mesh = None
        self.lvproblem = None

class VariableViscosityStokesSolver():
    def set_BTWB_dicts(self):
        BBCTWB_dict = {} # These are of type PETSc.Mat
        BBCTW_dict = {} # These are of type PETSc.Mat
        mh = self.problem.mh
        case = self.case
        w = self.w
        gamma = self.gamma
        for level in range(self.problem.nref+1):
            levelmesh = mh[level]
            Wlevel = self.problem.get_W_mat(levelmesh,case,w)
            BBClevel = self.problem.get_B_mat(levelmesh)
            Wlevel *= gamma
            if level in BBCTW_dict:
                BBCTWlevel = BBClevel.transposeMatMult(Wlevel, 
                                                  result=BBCTW_dict[level])
            else:
                BBCTWlevel = BBClevel.transposeMatMult(Wlevel)
                BBCTW_dict[level] = BBCTWlevel
            if level in BBCTWB_dict:
                BBCTWBlevel = BBCTWlevel.matMult(BBClevel, 
                                                 result=BBCTWB_dict[level])
            else:
                BBCTWBlevel = BBCTWlevel.matMult(BBClevel)
                BBCTWB_dict[level] = BBCTWBlevel

        self.BBCTWB_dict = BBCTWB_dict
        self.BBCTW_dict = BBCTW_dict 
        
    def set_transfers(self, transfers=None):
        if transfers is None and self.solver_type == "almg" \
                             and self.problem.discretisation == "cg":
            lvsolver = self.lvsolver
            dim  = self.problem.dim
            quad = self.problem.quad
            nref = self.problem.nref
            gamma = self.gamma
            mu_transfer = self.problem.get_viscosity()
            asmbackend = self.asmbackend
            def BTWBcb(level):
                return self.BBCTWB_dict[level]
            def Acb(level):
                ctx = lvsolver._ctx
                if level == nref:
                    splitctx = ctx.split([(0,)])[0]
                    A = splitctx._jac
                    A.form = splitctx.J
                    A.M = ctx._jac.M[0, 0]
                    A.petscmat = ctx._jac.petscmat.getNestSubMatrix(0, 0)
                    return A
                else:
                    ksp = lvsolver.snes.ksp.pc.getFieldSplitSubKSP()[0]
                    ctx = get_appctx(ksp.pc.getMGSmoother(level).dm)
                    A = ctx._jac
                    A.form = ctx.J
                    A.petscmat = A.petscmat
                    return A
            V, Q = self.problem.get_functionspace(self.problem.mesh)
            tdim = self.problem.mesh.topological_dimension()
            # vtransfer = AlgebraicSchoeberlTransfer((mu_transfer, gamma), 
            #                  A_callback, BTWB_callback, tdim, 'uniform', 
            #                  backend='lu', hexmesh=(dim == 3 and quad))
            vtransfer = AlgebraicSchoeberlTransfer((mu_transfer, gamma), 
                                 Acb, BTWBcb, tdim, 'uniform',
                                 backend=asmbackend,
                                 hexmesh=(dim==3 and quad))
            qtransfer = NullTransfer()
            transfers = {V.ufl_element(): (vtransfer.prolong, 
                                           vtransfer.restrict, 
                                           inject),
                         Q.ufl_element(): (prolong, restrict, qtransfer.inject)}
            transfermanager = TransferManager(native_transfers=transfers)
            self.transfers = transfers
            self.lvsolver.set_transfer_manager(transfermanager)
        else:
            self.transfers = transfers
            transfermanager = TransferManager(native_transfers=transfers)
            self.lvsolver.set_transfer_manager(transfermanager)

    def get_transfers(self):
        return self.transfers

    def set_nsp(self, nsp=None):
        mesh = self.problem.mesh
        V, Q = self.problem.get_functionspace(mesh)
        Z = V*Q
        if nsp is None:
            self.nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), 
                                        VectorSpaceBasis(constant=True)]) 
        else:
            self.nsp = nsp

    def set_parameters(self, params=None):
        dim  = self.problem.dim
        quad = self.problem.quad
        asmbackend = self.asmbackend
        solver_type = self.solver_type
        if params is None:
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

            mg_levels_solver_rich = {
                "ksp_type": "richardson",
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
                "pc_python_type": "hexstar.ASMHexStarPC" if (dim==3 and quad==True) 
                                                        else "firedrake.ASMStarPC",
                "pc_star_construct_dim": 0,
                "pc_star_backend": asmbackend,
                # "pc_star_sub_pc_asm_sub_mat_type": "seqaij",
                # "pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
                "pc_star_sub_sub_pc_factor_in_place": None,
                "pc_hexstar_construct_dim": 0,
                "pc_hexstar_backend": asmbackend,
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
                #"mg_levels": mg_levels_solver,
                "mg_coarse_pc_type": "python",
                "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                "mg_coarse_assembled_pc_type": "lu",
                "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
            }

            params = {
                "snes_type": "ksponly",
                #"snes_monitor": None,
                "mat_type": "nest",
                "ksp_type": "fgmres",
                "ksp_gmres_restart": 100,
                "ksp_rtol": 1.0e-6,
                "ksp_atol": 1.0e-10,
                "ksp_max_it": 1000,
                #"ksp_view": None,
                #"ksp_monitor_true_residual": None,
                #"ksp_converged_reason": None,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_factorization_type": "full",
                "pc_fieldsplit_schur_precondition": "user",
                "fieldsplit_1": fieldsplit_1,
            }

            if solver_type == "almg":
                fieldsplit_0_mg["mg_levels"] = mg_levels_solver
                params["fieldsplit_0"] = fieldsplit_0_mg
            elif solver_type == "almgcheb":
                fieldsplit_0_mg["mg_levels"] = mg_levels_solver_cheb
                params["fieldsplit_0"] = fieldsplit_0_mg
            elif solver_type == "almgrich":
                fieldsplit_0_mg["mg_levels"] = mg_levels_solver_rich
                params["fieldsplit_0"] = fieldsplit_0_mg
            elif solver_type == "allu":
                params["fieldsplit_0"] = fieldsplit_0_lu
            elif solver_type == "alamg":
                params["fieldsplit_0"] = fieldsplit_0_hypre
            elif solver_type == "lu":
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
                raise ValueError("please specify almg, allu or alamg for \
                                 --solver-type")
            self.params = params
        else:
            self.params = params

    def get_parameters(self):
        return self.params

    def set_linearvariationalsolver(self,
                                    augtopleftblock=True,
                                    modifyresidual=True):
        params = self.params
        nsp = self.nsp
        mesh = self.problem.mesh
        V, Q = self.problem.get_functionspace(mesh)
        Z = V*Q
        mu = self.problem.mu_fun(mesh)
        deg = self.problem.quaddeg
        dr = self.problem.mu_max/self.problem.mu_min
        nref = self.problem.nref
        appctx = {"nu_expr": mu, "gamma": self.gamma, 
                  "dr":dr, "case":self.case, "w":self.w, "deg":deg}

        def aug_topleftblock(X, J, ctx):
            if augtopleftblock is not True:
                return
            mh, level = get_level(ctx._x.ufl_domain())
            BTWBlevel = self.BBCTWB_dict[level]
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
            if modifyresidual is not True:
                return
            vel_is = Z._ises[0]
            pre_is = Z._ises[1]
            Fvel = F.getSubVector(vel_is)
            Fpre = F.getSubVector(pre_is)
            BTW  = self.BBCTW_dict[nref]
            Fvel += BTW*Fpre
            F.restoreSubVector(vel_is, Fvel)
            F.restoreSubVector(pre_is, Fpre)

        solver = LinearVariationalSolver(self.problem.lvproblem,
                                     solver_parameters=params,
                                     options_prefix="ns_",
                                     post_jacobian_callback=aug_topleftblock,
                                     post_function_callback=modify_residual,
                                     appctx=appctx, nullspace=nsp)
        self.lvsolver = solver
        self.set_transfers()
        self.Z = Z

    def get_iterationnum(self):
        return self.lvsolver.snes.ksp.getIterationNumber()

    def solve(self):
        self.lvsolver.solve()
        
    def __init__(self, problem, solver_type, case, gamma, asmbackend=None, w=0,
                 setBTWBdics=True):
        self.case = case
        self.w = w
        self.problem = problem
        self.solver_type = solver_type
        self.asmbackend = asmbackend
        self.gamma = Constant(gamma)
        self.params = None
        self.nsp = None
        self.transfers = None
        self.BBCTWB_dict = None
        self.BBCTW_dict = None
        self.lvsolver = None

        ## default setup
        if setBTWBdics is True:
            self.set_BTWB_dicts()
        self.set_parameters()

