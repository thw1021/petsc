# test changing stdout

from io import StringIO
import sys

sys.stdout = StringIO()
sys.stdout.write("This is the content of sys.stdout:\n")
sys.__stdout__.write("This is the content of sys.__stdout__:\n")

import numpy as np

from petsc4py import PETSc

a = np.random.rand(4)
a_vec = PETSc.Vec().createWithArray(a)
a_vec.view()
sys.__stdout__.write(sys.stdout.getvalue())
