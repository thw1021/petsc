from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------


class TestDevice(unittest.TestCase):
    def testCurrent(self):
        dctx = PETSc.DeviceContext().getCurrent()
        device = dctx.getDevice()
        del device
        del dctx
        dctx = PETSc.DeviceContext().getCurrent()
        device = dctx.getDevice()
        del device
        del dctx

    def testDevice(self):
        device = PETSc.Device.create()
        device.configure()
        _ = device.getDeviceType()
        _ = device.getDeviceId()
        del device

    def testDeviceContext(self):
        dctx = PETSc.DeviceContext().create()
        dctx.setUp()
        self.assertTrue(dctx.idle())
        dctx.destroy()

    def testStream(self):
        dctx = PETSc.DeviceContext().getCurrent()
        stype = dctx.getStreamType()
        dctx.setStreamType(stype)
        dctx.destroy()

    def testSetFromOptions(self):
        dctx = PETSc.DeviceContext().create()
        dctx.setFromOptions()
        dctx.setUp()
        dctx.destroy()

    def testDuplicate(self):
        dctx = PETSc.DeviceContext().getCurrent()
        dctx2 = dctx.duplicate()
        dctx.destroy()
        dctx2.destroy()

    def testWaitFor(self):
        dctx = PETSc.DeviceContext().create()
        dctx.setUp()
        dctx2 = PETSc.DeviceContext().create()
        dctx2.setUp()
        dctx.waitFor(dctx2)
        dctx.destroy()
        dctx2.destroy()

    def testForkJoin(self):
        dctx = PETSc.DeviceContext().getCurrent()
        jdestroy = PETSc.DeviceContext.JoinMode.DESTROY
        jtypes = [
            PETSc.DeviceContext.JoinMode.SYNC,
            PETSc.DeviceContext.JoinMode.NO_SYNC,
        ]
        for j in jtypes:
            dctxs = dctx.fork(2)
            dctx.join(j, dctxs)
            dctx.join(jdestroy, dctxs)
        dctx.destroy()


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
