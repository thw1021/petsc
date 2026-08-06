from petsc4py import PETSc
import unittest

class BaseTestPC:
    KSP_TYPE = None
    PC_TYPE = None
    def setUp(self):
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_SELF)
        pc = ksp.getPC()
        if self.KSP_TYPE:
            ksp.setType(self.KSP_TYPE)
        if self.PC_TYPE:
            pc.setType(self.PC_TYPE)
        self.ksp = ksp
        self.pc = pc

    def testAmatDef(self):
        pc_type = self.PC_TYPE
        for test_pc_type in ['mg']:
            pc = PETSc.PC().create()
            new_pc_type = 'none' if pc_type == test_pc_type else test_pc_type
            amat = []
            pc.setType(pc_type)
            amat.append(pc.getUseAmat())
            pc.setType(new_pc_type)
            amat.append(pc.getUseAmat())
            pc.setType(pc_type)
            amat.append(pc.getUseAmat())
            self.assertTrue(amat[0] == amat[2])
            self.assertFalse(amat[0] == amat[1])
            pc.destroy()

    def tearDown(self):
        self.ksp = None
        self.pc = None
        PETSc.garbage_cleanup()

class TestFIELDSPLITPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.FIELDSPLIT

    def testISoperations(self):
        test_index = [0,1,2]
        pc = self.pc
        is_u = PETSc.IS().createGeneral(test_index, comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(("u", is_u))

        self.assertTrue((pc.getFieldSplitSubIS("u").getIndices() == test_index).all())
        is_u = None


class TestMG(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.MG


class TestASMPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.ASM

    def testLocalSubdomains(self):
        pc = self.pc
        indices = [[0, 1, 2], [3, 4, 5]]
        is_sub = [PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF)
                  for idx in indices]
        pc.setASMLocalSubdomains(len(is_sub), is_sub)

        got_sub, got_local = pc.getASMLocalSubdomains()
        self.assertEqual(len(got_sub), len(is_sub))
        self.assertEqual(len(got_local), 0)
        for got, idx in zip(got_sub, indices):
            self.assertTrue((got.getIndices() == idx).all())
        for is in is_sub:
            is.destroy()

    def testLocalSubdomainsWithLocalPart(self):
        pc = self.pc
        indices = [[0, 1, 2, 3], [2, 3, 4, 5]]
        local_indices = [[0, 1], [4, 5]]
        is_sub = [PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF)
                  for idx in indices]
        is_local = [PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF)
                    for idx in local_indices]
        pc.setASMLocalSubdomains(len(is_sub), is_sub, is_local)

        got_sub, got_local = pc.getASMLocalSubdomains()
        self.assertEqual(len(got_sub), len(is_sub))
        self.assertEqual(len(got_local), len(is_local))
        for got, idx in zip(got_sub, indices):
            self.assertTrue((got.getIndices() == idx).all())
        for got, idx in zip(got_local, local_indices):
            self.assertTrue((got.getIndices() == idx).all())

        got_sub = None
        got_local = None
        is_sub = None
        is_local = None


class TestASMPCWorld(unittest.TestCase):

    def setUp(self):
        self.pc = PETSc.PC().create(PETSc.COMM_WORLD)
        self.pc.setType(PETSc.PC.Type.ASM)

    def tearDown(self):
        self.pc.destroy()
        self.pc = None
        PETSc.garbage_cleanup()

    def testLocalSubdomains(self):
        # The index sets are in the global numbering of the vector, so give
        # each process a distinct block of it.
        rank = PETSc.COMM_WORLD.getRank()
        indices = [3 * rank, 3 * rank + 1, 3 * rank + 2]
        is_sub = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
        self.pc.setASMLocalSubdomains(1, [is_sub])

        got_sub, got_local = self.pc.getASMLocalSubdomains()
        self.assertEqual(len(got_sub), 1)
        self.assertTrue((got_sub[0].getIndices() == indices).all())
        self.assertEqual(len(got_local), 0)

        got_sub = None
        got_local = None
        is_sub = None

    def testLocalSubdomainsUnset(self):
        # Nothing has been set and the preconditioner has not been set up, so
        # PETSc holds no subdomains yet.
        got_sub, got_local = self.pc.getASMLocalSubdomains()
        self.assertEqual(got_sub, [])
        self.assertEqual(got_local, [])


if __name__ == '__main__':
    unittest.main()
