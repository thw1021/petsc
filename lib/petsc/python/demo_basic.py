import petsc
import numpy as np

# The PETSc python binding has three models of interaction

########## 1)  Call petsc.solve(), providing the needed data as arguments

solution = petsc.solve(rhs = np.array([1,1]), matrix = np.array([[1, 2], [3, 4]]))

########## 2) Instantiate a PETSc.Problem class, providing the needed data as arguments (for repeated related solves)

ls = petsc.LinearSystem(matrix = np.array([[1, 2], [3, 4]]))
solution = ls.solve(rhs = np.array([1,2]))
solution = ls.solve(rhs = np.array([2,3]))

########## 3) Instantiate a user class inherited from a petsc.Problem class and (optional) PETSc.Grid class (see demo_grid.py)

class LinearSystem(petsc.LinearSystem):
  def createMatrix(self,**kwargs):
    return np.array([[1, 2], [3, 4]])

  def evalRHS(self,rhs):
    rhs[0] = 1
    rhs[1] = 2

ls = LinearSystem(comm = petsc.PETSc.COMM_SELF)
solution = ls.solve(ksp_monitor='')

########## Vector and matrix arguments can be numpy arrays or PETSc.Vec and Petsc.Mat

solution = petsc.solve(rhs = petsc.PETSc.Vec().createWithArray(np.array([1,1])), matrix = np.array([[1, 2], [3, 4]]))

########## Call petsc.solve() (nonlinear), providing the needed data as arguments (petsc.solve() handles linear and nonlinear systems)

def evalFunction(x, f):
  f[0] = (x[0]*x[0] + x[0]*x[1] - 3.0).item()
  f[1] = (x[0]*x[1] + x[1]*x[1] - 6.0).item()

def evalJacobian(x, J):
  J[0,0] = (2.0*x[0] + x[1]).item()
  J[0,1] = (x[0]).item()
  J[1,0] = (x[1]).item()
  J[1,1] = (x[0] + 2.0*x[1]).item()

solution = petsc.solve(solution = np.array([2, 3]),jacobian = np.array([[1, 2], [3, 4]]),evalFunction = evalFunction,
                       evalJacobian = evalJacobian)

########## Instantiate a user class inherited from petsc.NonlinearSystem

class NonlinearSystem(petsc.NonlinearSystem):
  def createJacobian(self,**kwargs):
    mat = petsc.PETSc.Mat().createDense((2,2),comm = self.comm).setUp()
    return mat

  def evalFunction(self, x, f):
    f[0] = (x[0]*x[0] + x[0]*x[1] - 3.0).item()
    f[1] = (x[0]*x[1] + x[1]*x[1] - 6.0).item()

  def evalJacobian(self, x, J):
    J[0,0] = (2.0*x[0] + x[1]).item()
    J[0,1] = (x[0]).item()
    J[1,0] = (x[1]).item()
    J[1,1] = (x[0] + 2.0*x[1]).item()

nls = NonlinearSystem(comm = petsc.PETSc.COMM_SELF)
solution = nls.solve(solution=np.array([2, 3]),nls_monitor='',nls_view='')

########## Call petsc.integrate(), providing the needed data as arguments

def createIJacobian(comm = petsc.COMM_SELF,**kwargs):
  mat = petsc.PETSc.Mat().createDense((3,3),comm = comm).setUp()
  return mat

def evalIFunction(t,u,du,F):
  f = du + u * u
  f.copy(F)

def evalIJacobian(t,u,du,a,J):
  diag = a + 2 * u
  J.setDiagonal(diag)

solution = petsc.integrate(solution=np.array([1, 2, 3]),evalIFunction = evalIFunction,createIJacobian = createIJacobian,
                           evalIJacobian = evalIJacobian, ode_max_time=1,ode_monitor='',ode_view='')

########## Instantiate a user class inherited from petsc.ODE

class ODE(petsc.ODE):
  def createIJacobian(self,**kwargs):
    mat = petsc.PETSc.Mat().createDense((3,3),comm = self.comm).setUp()
    return mat

  def evalIFunction(self, t,u,du,F):
    f = du + u * u
    f.copy(F)

  def evalIJacobian(self,t,u,du,a,J):
    diag = a + 2 * u
    J.setDiagonal(diag)

ode = ODE()
solution = ode.integrate(solution==np.array([1, 2, 3]),ode_max_time=1,ode_monitor='',ode_view='')

########## Call petsc.optimize(), providing the needed data as arguments

def evalObjective(x):
  return (x[0] - 2.0)**2 + (x[1] - 2.0)**2 - 2.0*(x[0] + x[1])

def evalGradient(x, g):
  g[0] = 2.0*(x[0] - 2.0) - 2.0
  g[1] = 2.0*(x[1] - 2.0) - 2.0

solution = petsc.optimize(solution=np.array([0,0]),evalObjective = evalObjective, evalGradient = evalGradient, opt_monitor='',
                          opt_view='',opt_view_solution='')

########## Instantiate a user class inherited from petsc.Optimization

class Optimization(petsc.Optimization):
  def createLeftVector(self,**kwargs):
    vec = petsc.PETSc.Vec().create()
    vec.setSizes(2) # TODO: fix petsc4py to allow chaining these
    vec.setUp()
    return vec

  def createRightVector(self,**kwargs):
    return self.createLeftVector(**kwargs)

  def evalObjective(self, x):
    return (x[0] - 2.0)**2 + (x[1] - 2.0)**2 - 2.0*(x[0] + x[1])

  def evalGradient(self, x, g):
    g[0] = 2.0*(x[0] - 2.0) - 2.0
    g[1] = 2.0*(x[1] - 2.0) - 2.0

opt = Optimization()
solution = opt.optimize(opt_monitor='',opt_view='',opt_view_solution='')
   # bug somewhere, tao_view does not work if provided here but tao_monitor does?
   # bug in Tao, view_solution acts like monitor_solution

