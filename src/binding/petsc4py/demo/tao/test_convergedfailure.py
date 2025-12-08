# Tao handling of convergence test
# =============================
#
# Use the petsc4py interface to Tao to demonstate setObjectiveDomainError

import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc

class AppCtx(object):

    """
    Bogus callbacks needed by Tao
    """

    def __init__(self):
        '''On which call to the functions will it return a domain error'''
        self.domainerrorcnt = 1;
        self.cnt = 0
        pass

    def formObjective(self, tao, x):
        if self.cnt == self.domainerrorcnt: tao.setObjectiveDomainError()
        self.cnt += 1
        return (x[0] - 1)*(x[0] - 1)

    def formGradient(self, tao, x, G):
        G[0] = 2*(x[0] - 1)

    def formObjGrad(self, tao, x, G):
        G[0] = 2*(x[0] - 1)
        if self.cnt == self.domainerrorcnt: tao.setObjectiveDomainError()
        self.cnt += 1
        return  (x[0] - 1)*(x[0] - 1)

ctx = AppCtx()

x = PETSc.Vec().create(PETSc.COMM_SELF)
x.setSizes(1)
x.setFromOptions()

tao = PETSc.TAO().create(comm=PETSc.COMM_WORLD)
# tao.setErrorIfNotConverged(True) or -tao_error_if_not_converged
tao.setFromOptions()
tao.setObjectiveGradient(ctx.formObjGrad)
tao.setObjective(ctx.formObjective)
tao.setGradient(ctx.formGradient)
tao.setSolution(x)
tao.solve()




