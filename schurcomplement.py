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

        self.viscmassinv = None
        self.massinv = None

        ## Case 1:
        # massinv = assemble(Tensor(inner(u, v)*dx).inv)
        # self.massinv = massinv.petscmat
        # self.scale = nu.copy(deepcopy=True)
        # self.scale.project(-(nu+gamma))

        ## Case 2:
        # massinv = assemble(Tensor(inner(u, v)*dx).inv)
        # self.massinv = massinv.petscmat
        # self.scale = nu.copy(deepcopy=True)
        # self.scale.project(-(sqrt(dr)+gamma))

        ## Case 3:
        viscmassinv = assemble(Tensor(-1.0/nu*inner(u, v)*dx).inv)
        massinv = assemble(Tensor(inner(u, v)*dx).inv)
        self.viscmassinv = viscmassinv.petscmat
        self.massinv = massinv.petscmat
        self.scale = nu.copy(deepcopy=True)
        self.scale.project(-gamma)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        tmp = y.duplicate()
        self.massinv.mult(x, tmp)
        with self.scale.dat.vec_wo as w:
            tmp.pointwiseMult(tmp, w)
        if self.viscmassinv is not None:
            self.viscmassinv.multAdd(x,tmp,y)
        else:
            tmp.copy(y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")
