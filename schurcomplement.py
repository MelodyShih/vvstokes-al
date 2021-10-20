from firedrake import *

class DGMassInv(PCBase):

    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)

        nu_exprlist = appctx["nu_exprlist"]
        dxlist = appctx["dxlist"]
        assert len(nu_exprlist) == len(dxlist)

        nu_expr = nu_exprlist[0]
        dx      = dxlist[0]
        gamma   = appctx["gamma"]
        dr      = appctx["dr"]
        case    = appctx["case"]
        w       = appctx["w"]
        deg     = appctx["deg"]

        self.viscmassinv = None
        self.massinv = None
        self.case = case
        self.w = w
        self.gamma = gamma

        if case == 0:
            massinv = assemble(Tensor(inner(u, v)*dx(degree=deg)).inv)
            self.massinv = massinv.petscmat
            self.scale = nu_fun.copy(deepcopy=True)
            self.scale = Function(V).project(-(1.0+gamma))
        elif case == 1:
            massinv = assemble(Tensor(inner(u, v)*dx(degree=deg)).inv)
            self.massinv = massinv.petscmat
            nu_fun = Function(V).interpolate(nu_expr)
            self.scale.project(-(nu_fun+gamma))
        elif case == 2:
            massinv = assemble(Tensor(inner(u, v)*dx(degree=deg)).inv)
            self.massinv = massinv.petscmat
            self.scale = Function(V).project(-(sqrt(dr)+gamma))
        elif case == 3 or case == 4:
            weak = -1.0/nu_expr*inner(u,v)*dx(degree=deg)
            viscmassinv = assemble(Tensor(weak).inv)

            massinv = assemble(Tensor(inner(u, v)*dx(degree=deg)).inv)
            self.viscmassinv = viscmassinv.petscmat
            self.massinv = massinv.petscmat
            self.scale = Function(V).project(-gamma)
        elif case == 5:
            viscmassinv = assemble(Tensor(-1.0/nu_expr*inner(u, v)*\
                                                           dx(degree=deg)).inv)
            self.viscmassinv = viscmassinv.petscmat
            self.scale = Function(V).project(1.0+gamma)
        elif case == 6:
            ## P = Shat - gamma*W
            # Shat = -viscmassinv
            # W = w*viscmassinv + (1-w)*massinv
            massinv     = assemble(Tensor(inner(u, v)*dx).inv)
            viscmassinv = assemble(Tensor(1.0/nu_expr*inner(u, v)*\
                                                           dx(degree=deg)).inv)
            self.viscmassinv = viscmassinv.petscmat
            self.massinv     = massinv.petscmat
            self.scale = Function(V).project(-gamma)
            self.w = w
        else:
            raise ValueError("Unknown type of preconditioner %i" % case)
    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        if self.case <= 4:
            # Case 1,2,3,4:
            tmp = y.duplicate()
            self.massinv.mult(x, tmp)
            with self.scale.dat.vec_wo as w:
                tmp.pointwiseMult(tmp, w)
            if self.viscmassinv is not None:
                self.viscmassinv.multAdd(x,tmp,y)
            else:
                tmp.copy(y)
        elif self.case == 5:
            ## Case 5:
            self.viscmassinv.mult(x, y)
            with self.scale.dat.vec_wo as w:
                y.pointwiseMult(y, w)
        elif self.case == 6:
            tmp = y.duplicate()
            self.massinv.mult(x, tmp)
            tmp = tmp*(1-self.w)
            tmp2 = y.duplicate()
            self.viscmassinv.mult(x, tmp2)
            tmp2 = tmp2*(self.w)
            tmp3 = tmp + tmp2
            with self.scale.dat.vec_wo as w:
                tmp3.pointwiseMult(tmp3, w)
            self.viscmassinv.multAdd(-x,tmp3,y)
        else:
            raise ValueError("Unknown type of preconditioner %i" % case)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")

    def destroy(self, pc):
        if self.viscmassinv is not None:
            #PETSc.Sys.Print("[mem] local viscmassinv size",
            #                  self.viscmassinv.getInfo(1)['memory'])
            self.viscmassinv.destroy()
        if self.massinv is not None:
            #PETSc.Sys.Print("[mem] local massinv size",
            #                  self.massinv.getInfo(1)['memory'])
            self.massinv.destroy()
