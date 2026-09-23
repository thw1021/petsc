import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc


# The user-defined Python class implementing the gradient descent.
class myGradientDescent:
    def create(self, tao):
        # Create a line search type with constant step size.
        self._ls = PETSc.TAOLineSearch().create(comm=PETSc.COMM_SELF)
        self._ls.useTAORoutine(tao)
        self._ls.setType(PETSc.TAOLineSearch.Type.UNIT)
        self._ls.setInitialStepLength(0.2)

    def solve(self, tao):
        # Get solution and Jacobian vector.
        x = tao.getSolution()
        gradient = tao.getGradient()[0]

        # Prepare search direction for line search.
        search_direction = gradient.copy()

        # Optimization loop.
        for it in range(tao.getMaximumIterations()):
            tao.setIterationNumber(it)

            # Compute search_direction.
            #   search_direction = -gradient
            tao.computeGradient(x, gradient)
            gradient.copy(search_direction)
            search_direction.scale(-1)

            # Apply line search:
            #   x += .2 search_direction
            f, _, reason = self._ls.apply(x, gradient, search_direction)

            if reason < 0:
                raise RuntimeError('LS failed.')

            # Log and update internal state.
            tao.monitor(f=f, res=gradient.norm())

            # Convergence check.
            if tao.checkConverged() > 0:
                break


# Minimize f(x) = (x[0] - 1)^2 + (x[1] - 2)^2.
def objective(tao, x):
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def gradient(tao, x, g):
    g[0] = 2.0 * (x[0] - 1.0)
    g[1] = 2.0 * (x[1] - 2.0)
    g.assemble()


# Create the initial guess and gradient vector.
x = PETSc.Vec().createSeq(2, comm=PETSc.COMM_SELF)
x.set(0.5)
g = x.duplicate()

# Create the Python optimizer and configure the minimization.
tao = PETSc.TAO().createPython(myGradientDescent(), comm=PETSc.COMM_SELF)
tao.setObjective(objective)
tao.setGradient(gradient, g)
tao.setSolution(x)
tao.setTolerances(gatol=1e-6)
tao.setMaximumIterations(100)
tao.setFromOptions()
tao.solve()

PETSc.Sys.Print(f'x = ({x[0]:.6f}, {x[1]:.6f})')
