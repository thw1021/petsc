import sys
import petsc4py
import pywt
import math
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np

def soft(vecin, vecout, thresh):
    start, end = vecin.getOwnershipRange()
    for i in range(start,end):
        vecout[i] = (vecin[i]/np.abs(vecin[i])) * np.maximum(np.abs(vecin[i]) - thresh, 0)
    vecin.assemble()
    vecout.assemble()


class MyDMTao:
    def __init__(self):
        self.log = {}

    def _log(self, method):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, dm):
        self._log('create')
        self.testvec = PETSc.Vec()

    def destroy(self, dm):
        self._log('destroy')
        self.testvec.destroy()

    def setFromOptions(self, dm):
        self._log('setFromOptions')

    def setUp(self, dm):
        self._log('setUp')
        self.testvec = dm.getSolution().duplicate()

    def applyproximalmap(self, dm0, dm1, thresh, vecin, vecout, flg):
        self._log('applyproximalmap')
        start, end = vecin.getOwnershipRange()
        for i in range(start,end):
            vecout[i] = (vecin[i]/np.abs(vecin[i])) * np.maximum(np.abs(vecin[i]) - thresh, 0)
        vecin.assemble()
        vecout.assemble()

#Build-in DMTAO
OptDB = PETSc.Options()

n = OptDB.getInt('n', 10)
thresh  = OptDB.getReal('lambda', 0.1)

dm0 = PETSc.DM()
dm0.create(PETSc.COMM_WORLD)
dm0.setTAOType(PETSc.DM.TAOType.L1)

dm1 = PETSc.DM()
dm1.create(PETSc.COMM_WORLD)
dm1.setTAOType(PETSc.DM.TAOType.L2)

y1 = PETSc.Vec()
y1.create(PETSc.COMM_WORLD)
y1.setSizes(n)
y1.setType(PETSc.Vec.Type.SEQ)
y1.setRandom()
y2 = y1.duplicate()
y1.copy(y2)

x1 = PETSc.Vec()
x1.create(PETSc.COMM_WORLD)
x1.setSizes(n)
x1.setType(PETSc.Vec.Type.SEQ)
x1.set(0)
x2 = x1.duplicate()
x1.copy(x2)

dm0.applyTAOproximalmap(dm1, thresh, y1, x1, False)

# DMTAOPYTHON
dmpy0 = PETSc.DM()
dmpy0.create(PETSc.COMM_WORLD)
dmpy0.createTAOPython(MyDMTao(), comm=PETSc.COMM_WORLD)
dmpy0.applyTAOproximalmap(dm1, thresh,  y2, x2, False)

y1.viewFromOptions('-view_input_builtin')
y2.viewFromOptions('-view_input_py')
x1.viewFromOptions('-view_sol_builtin')
x2.viewFromOptions('-view_sol_py')

test = x1.equal(x2)
print(test)
