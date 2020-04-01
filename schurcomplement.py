from firedrake import *

class DGMassInv(PCBase):

    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)
        nu = appctx["nu"]
        gamma = appctx["gamma"]
        massinv = assemble(Tensor(inner(u, v)*dx).inv)
        self.massinv = massinv.petscmat
        self.nuplusgammainv = nu.copy(deepcopy=True)
        self.nuplusgammainv.project(-(nu+gamma))


    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        with self.nuplusgammainv.dat.vec_wo as w:
            y.pointwiseMult(y, w)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")

