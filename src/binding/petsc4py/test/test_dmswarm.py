from petsc4py import PETSc
import unittest

import numpy as np


class TestDMSwarm(unittest.TestCase):
    def setUp(self):
        swarm = PETSc.DMSwarm().create(PETSc.COMM_SELF)
        swarm.setType(PETSc.DMSwarm.Type.BASIC)
        swarm.initializeFieldRegister()
        swarm.registerField('field_a', 1, PETSc.RealType)
        swarm.registerField('field_b', 1, PETSc.RealType)
        swarm.registerField('coordinates', 2, PETSc.RealType)
        swarm.registerField('weights', 1, PETSc.RealType)
        swarm.finalizeFieldRegister()
        swarm.setLocalSizes(2, 0)
        self.swarm = swarm

    def tearDown(self):
        self.swarm.destroy()
        self.swarm = None
        PETSc.garbage_cleanup()

    def setField(self, name, values):
        field = self.swarm.getField(name)
        field[...] = values
        self.swarm.restoreField(name)

    def getField(self, name):
        field = self.swarm.getField(name)
        values = field.copy()
        self.swarm.restoreField(name)
        return values

    def testVectorFromField(self):
        for scope in ('Global', 'Local'):
            with self.subTest(scope=scope):
                create = getattr(self.swarm, f'create{scope}VectorFromField')
                destroy = getattr(self.swarm, f'destroy{scope}VectorFromField')
                self.setField('field_a', [[1], [2]])
                vec = create('field_a')
                vec.array[:] = [3, 4]
                destroy('field_a', vec)
                self.assertFalse(vec)
                np.testing.assert_array_equal(self.getField('field_a').ravel(), [3, 4])

    def testVectorFromFields(self):
        names = ['field_a', 'field_b']
        for scope in ('Global', 'Local'):
            with self.subTest(scope=scope):
                create = getattr(self.swarm, f'create{scope}VectorFromFields')
                destroy = getattr(self.swarm, f'destroy{scope}VectorFromFields')
                self.setField('field_a', [[1], [2]])
                self.setField('field_b', [[3], [4]])
                vec = create(names)
                np.testing.assert_array_equal(vec.array, [1, 3, 2, 4])
                vec.array[:] = [5, 7, 6, 8]
                destroy(names, vec)
                self.assertFalse(vec)
                np.testing.assert_array_equal(self.getField('field_a').ravel(), [5, 6])
                np.testing.assert_array_equal(self.getField('field_b').ravel(), [7, 8])

    def testComputeMoments(self):
        self.setField('coordinates', [[1, 2], [3, 4]])
        self.setField('weights', [[2], [5]])
        moments = self.swarm.computeMoments('coordinates', 'weights')
        np.testing.assert_array_equal(moments, [7, 17, 24, 135])


class TestCellDM(unittest.TestCase):
    def testCreateAndLookup(self):
        dm = PETSc.DM().create(PETSc.COMM_SELF)
        dm.setName('cell_dm')
        cell = PETSc.CellDM().create(dm, ['field_a'], ['coordinate_a', 'coordinate_b'])
        self.assertEqual(cell.getFields(), ['field_a'])
        self.assertEqual(cell.getCoordinateFields(), ['coordinate_a', 'coordinate_b'])

        swarm = PETSc.DMSwarm().create(PETSc.COMM_SELF)
        swarm.setDimension(2)
        swarm.addCellDM(cell)
        swarm.setCellDMActive('cell_dm')
        found = swarm.getCellDMByName('cell_dm')
        self.assertEqual(found.getFields(), ['field_a'])

        found.destroy()
        swarm.destroy()
        cell.destroy()
        dm.destroy()
        PETSc.garbage_cleanup()


class TestDMSwarmSort(unittest.TestCase):
    def testGetPointsPerCellRestoresWorkArray(self):
        plex = PETSc.DMPlex().createFromCellList(
            2,
            [[0, 1, 2]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            comm=PETSc.COMM_SELF,
        )
        swarm = PETSc.DMSwarm().create(PETSc.COMM_SELF)
        swarm.setDimension(2)
        swarm.setType(PETSc.DMSwarm.Type.PIC)
        swarm.setCellDM(plex)
        swarm.finalizeFieldRegister()
        swarm.setLocalSizes(1, 0)
        cStart, _ = plex.getHeightStratum(0)
        cell_dm = swarm.getCellDMActive()
        cell_id = cell_dm.getCellID()
        cell_dm.destroy()
        field = swarm.getField(cell_id)
        field[0] = cStart
        swarm.restoreField(cell_id)
        swarm.sortGetAccess()
        try:
            first = swarm.sortGetPointsPerCell(cStart)
            second = swarm.sortGetPointsPerCell(cStart)
        finally:
            swarm.sortRestoreAccess()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        swarm.destroy()
        plex.destroy()
        PETSc.garbage_cleanup()


if __name__ == '__main__':
    unittest.main()
