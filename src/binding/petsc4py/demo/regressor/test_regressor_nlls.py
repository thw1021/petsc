# Nonlinear least squares regression test
# =======================================
#
# Use the petsc4py interface to PetscRegressor to fit the exponential model
# y = p0 * exp(p1 * x) + p2, the same problem solved in
# src/ml/regressor/tests/ex_nlls.c.

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy
from petsc4py import PETSc


# The rows of X, and hence of the model output, are distributed over the
# communicator, but every process needs the whole parameter vector to evaluate
# the model. The scatter and the sequential vector holding the replicated
# parameters are built once and handed to the callbacks as extra arguments.
def gather_parameters(p, scatter, p_all):
    scatter.scatter(p, p_all, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    return p_all.getArray(readonly=True)


def model(regressor, X, p, f, scatter, p_all):
    """Evaluate f(X, p) = p0 * exp(p1 * x) + p2."""
    pv = gather_parameters(p, scatter, p_all)
    x = X.getDenseArray()[:, 0]
    f.setArray(pv[0] * numpy.exp(pv[1] * x) + pv[2])


def jacobian(regressor, X, p, J, P, scatter, p_all):
    """Evaluate the derivative of the model with respect to the parameters."""
    pv = gather_parameters(p, scatter, p_all)
    x = X.getDenseArray()[:, 0]
    rstart, rend = J.getOwnershipRange()
    e = numpy.exp(pv[1] * x)
    values = numpy.empty((rend - rstart, 3))
    values[:, 0] = e
    values[:, 1] = pv[0] * x * e
    values[:, 2] = 1.0
    J.setValues(list(range(rstart, rend)), [0, 1, 2], values)
    J.assemble()
    if P != J:
        P.assemble()


def main():
    comm = PETSc.COMM_WORLD

    OptDB = PETSc.Options()
    use_analytic_jacobian = OptDB.getBool('use_analytic_jacobian', True)

    M = 20
    N = 3
    p_true = [2.5, -0.3, 0.5]
    p0_array = [1.0, 0.0, 0.0]

    # Build the M x 1 data matrix X (rows hold x_i in [0, 4]) and the target
    # y = f(x; p_true).
    X = PETSc.Mat().createDense((M, 1), comm=comm)
    X.setUp()
    y = X.createVecLeft()
    y_predicted = y.duplicate()

    rstart, rend = X.getOwnershipRange()
    x_local = numpy.linspace(0.0, 4.0, M)[rstart:rend]
    X.getDenseArray()[:, 0] = x_local
    y.setArray(p_true[0] * numpy.exp(p_true[1] * x_local) + p_true[2])
    X.assemble()

    # Initial parameter guess.
    p0 = PETSc.Vec().create(comm=comm)
    p0.setSizes(N)
    p0.setFromOptions()
    if not comm.getRank():
        p0.setValues(range(N), p0_array)
    p0.assemble()

    scatter, p_all = PETSc.Scatter.toAll(p0)

    regressor = PETSc.Regressor().create(comm=comm)
    regressor.setType(PETSc.Regressor.Type.NLLS)
    regressor.setRegularizerWeight(0.0)
    regressor.setFromOptions()
    regressor.setNLLSFunction(model, args=(scatter, p_all))
    regressor.setNLLSInitialParameters(p0)
    if use_analytic_jacobian:
        regressor.setNLLSJacobian(jacobian, args=(scatter, p_all))
    regressor.fit(X, y)
    p_fit = regressor.getNLLSParameters()
    regressor.predict(X, y_predicted)

    PETSc.Sys.Print('Fitted parameters:')
    p_fit.view()
    PETSc.Sys.Print('Predicted values:')
    y_predicted.view()

    scatter.destroy()
    p_all.destroy()
    X.destroy()
    y.destroy()
    y_predicted.destroy()
    p0.destroy()
    regressor.destroy()


if numpy.iscomplexobj(PETSc.ScalarType()):
    PETSc.Sys.Print('PETSCREGRESSORNLLS is not supported with complex scalars')
else:
    main()
