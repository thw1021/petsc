import unittest
from petsc4py import PETSc
from sys import getrefcount
import numpy as np


# --------------------------------------------------------------------


class MyDMTao:
    def __init__(self):
        self.log = {}

    def _log(self, method):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, dm):
        self._log('create')

    def destroy(self, dm):
        self._log('destroy')

    def setFromOptions(self, dm):
        self._log('setFromOptions')

    def setUp(self, dm):
        self._log('setUp')

    def applyproximalmap(self, dm0, dm1, thresh, vecin, vecout, flg):
        self._log('applyproximalmap')
        start, end = vecin.getOwnershipRange()
        for i in range(start,end):
            vecout[i] = (vecin[i]/np.abs(vecin[i])) * np.maximum(np.abs(vecin[i]) - thresh, 0)
        vecin.assemble()
        vecout.assemble()

class TestDMTaoPython(unittest.TestCase):
    def setUp(self):
        self.dm = PETSc.DM()
        self.dm.createTAOPython(MyDMTao(), comm=PETSc.COMM_SELF)
        ctx = self.dm.getTAOPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.assertEqual(ctx.log['create'], 1)

    def tearDown(self):
        ctx = self.dm.getTAOPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.assertTrue('destroy' not in ctx.log)
        self.dm.setTAOFromOptions()
        self.dm.destroy()
        self.dm = None
        PETSc.garbage_cleanup()
        self.assertEqual(ctx.log['destroy'], 1)
        self.assertEqual(getrefcount(ctx), 2)

    def testGetType(self):
        ctx = self.dm.getTAOPythonContext()
        pytype = f'{ctx.__module__}.{type(ctx).__name__}'
        self.assertTrue(self.dm.getTAOPythonType() == pytype)

    def testProx(self):
        thresh = 0.1
        dm     = self.dm
        x      = PETSc.Vec().create(dm.getComm())

        x.setType('standard')
        x.setSizes(10)
        x.set(0)
        y1 = x.duplicate()
        y1.setRandom()
        y2 = y1.duplicate()
        y1.copy(y2)

        x_l1_test = x.duplicate()
        x.copy(x_l1_test)

        dml1= PETSc.DM()
        dml1.create(dm.getComm())
        dml1.setTAOType(PETSc.DM.TAOType.L1)

        dml2 = PETSc.DM()
        dml2.create(dm.getComm())
        dml2.setTAOType(PETSc.DM.TAOType.L2)

        # Solve built-in version
        dml1.applyTAOproximalmap(dml2, thresh, y2, x_l1_test, False)

        # Solve MyDMTao version
        dm.applyTAOproximalmap(dml2, thresh, y1, x, False)

        self.assertTrue(x.equal(x_l1_test))
        dml1.destroy()
        dml2.destroy()
        x_l1_test.destroy()
        x.destroy()
        y1.destroy()
        y2.destroy()


# --------------------------------------------------------------------

if np.iscomplexobj(PETSc.ScalarType()):
    del TestDMTaoPython

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
