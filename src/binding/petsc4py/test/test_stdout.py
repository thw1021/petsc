import unittest

# --------------------------------------------------------------------


class TestStdout(unittest.TestCase):
    def testStdoutRedirect(self):
        from io import StringIO
        import sys
        prevstdout = sys.stdout
        sys.stdout = StringIO()

        import numpy as np
        from petsc4py import PETSc

        if not (__name__ == '__main__'):
            PETSc._push_python_stdout()

        a = np.array([0.,0.,0.])
        a_vec = PETSc.Vec().createWithArray(a,comm=PETSc.COMM_SELF)
        a_vec.view()
        newstdout = sys.stdout
        sys.stdout = prevstdout
        output = newstdout.getvalue()
        sys.stdout = prevstdout
        if not (__name__ == '__main__'):
            PETSc._pop_python_stdout()
        shouldbe = \
"""Vec Object:\x00 1 MPI processes
\x00  type: seq
\x000.
0.
0.
"""
        self.assertEqual(output,shouldbe)

        

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
