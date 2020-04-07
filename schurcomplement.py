from firedrake import *

class DGMassInv(PCBase):

    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)

        nu    = appctx["nu"]
        gamma = appctx["gamma"]
        dr    = appctx["dr"]
        case  = appctx["case"]

        self.viscmassinv = None
        self.massinv = None
        self.case = case

        if case == 1:
            massinv = assemble(Tensor(inner(u, v)*dx).inv)
            self.massinv = massinv.petscmat
            self.scale = nu.copy(deepcopy=True)
            self.scale.project(-(nu+gamma))
        elif case == 2:
            massinv = assemble(Tensor(inner(u, v)*dx).inv)
            self.massinv = massinv.petscmat
            self.scale = nu.copy(deepcopy=True)
            self.scale.project(-(sqrt(dr)+gamma))
        elif case == 3:
            viscmassinv = assemble(Tensor(-1.0/nu*inner(u, v)*dx).inv)
            massinv = assemble(Tensor(inner(u, v)*dx).inv)
            self.viscmassinv = viscmassinv.petscmat
            self.massinv = massinv.petscmat
            self.scale = nu.copy(deepcopy=True)
            self.scale.project(-gamma)
        elif case == 4:
            viscmassinv = assemble(Tensor(-1.0/nu*inner(u, v)*dx).inv)
            self.viscmassinv = viscmassinv.petscmat
            self.scale = nu.copy(deepcopy=True)
            self.scale.project(1.0+gamma)
        else:
            raise ValueError("Unknown type of preconditioner %i" % case)
    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        if self.case < 4:
            # Case 1,2,3:
            tmp = y.duplicate()
            self.massinv.mult(x, tmp)
            with self.scale.dat.vec_wo as w:
                tmp.pointwiseMult(tmp, w)
            if self.viscmassinv is not None:
                self.viscmassinv.multAdd(x,tmp,y)
            else:
                tmp.copy(y)
        elif self.case == 4:
            ## Case 4:
            self.viscmassinv.mult(x, y)
            with self.scale.dat.vec_wo as w:
                y.pointwiseMult(y, w)
        else:
            raise ValueError("Unknown type of preconditioner %i" % case)
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")
