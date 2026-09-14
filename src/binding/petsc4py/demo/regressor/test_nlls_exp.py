"""End-to-end NLLS binding smoke test: fit y = p0*exp(p1*x) + p2.

Mirrors `src/ml/regressor/tests/ex_nlls.c` (which is run serially: no `nsize`).
The model and Jacobian callbacks exercise the Python -> C -> Python trampoline
path, which acquires the GIL via `with gil`.

    PYTHONPATH=<petsc>/lib/petsc4py PETSC_ARCH=<arch> python3 test_nlls_exp.py
"""
import sys
sys.argv = [sys.argv[0]]
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

p_true = np.array([2.5, -0.3, 0.5])
M = 20

x = np.linspace(0.0, 4.0, M)
y = p_true[0] * np.exp(p_true[1] * x) + p_true[2]

rows = np.arange(M, dtype=np.int32)

# Data matrix X (M x 1, column 0 holds x_i) and target y.
X = PETSc.Mat().createDense(M, 1, comm=PETSc.COMM_WORLD)
yv = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
yv.setSizes(M)
yv.setFromOptions()
yv.setUp()
X.setValues(rows, [0], x[:, None])
yv.setValues(rows, y)
X.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
X.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
yv.assemblyBegin()
yv.assemblyEnd()


def model(reg, Xm, p, f):
    rstart, mend = Xm.getOwnershipRange()
    xa = 4.0 * np.arange(rstart, mend) / (M - 1)   # local x_i, as in ex_nlls.c
    pa = p.getArray(readonly=1)                    # TAO locks p read-only
    f.setArray(pa[0] * np.exp(pa[1] * xa) + pa[2])


def jacobian(reg, Xm, p, J, Jpre):
    rstart, mend = J.getOwnershipRange()
    xa = 4.0 * np.arange(rstart, mend) / (M - 1)
    pa = p.getArray(readonly=1)
    e = pa[0] * np.exp(pa[1] * xa)
    J.setValues(np.arange(mend - rstart, dtype=np.int32), [0, 1, 2],
                np.c_[e, e * xa, np.ones_like(xa)])
    J.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    J.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
    if Jpre is not J:
        Jpre.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        Jpre.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)


r = PETSc.Regressor().create(comm=PETSc.COMM_WORLD)
r.setType("nlls")
r.setFromOptions()
r.setNLLSFunction(model)
r.setNLLSJacobian(jacobian)
r.setNLLSInitialParameters(PETSc.Vec().createWithArray(np.array([1.0, 0.0, 0.0])))
r.fit(X, yv)

pfit = r.getNLLSParameters().getArray()
print("fit      =", pfit)
print("true     =", p_true)
print("max|err| =", np.max(np.abs(pfit - p_true)))
assert np.allclose(pfit, p_true, atol=1.0e-1), "fit did not recover parameters"
print("NLLS binding works")
