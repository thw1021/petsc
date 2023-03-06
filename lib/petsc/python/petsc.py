#
#  Prototype - Provides a Pythonic binding for PETSc/Tao/SLEPc on top of the pets4py binding
#
#  Goals:
#  * be more pythonic
#  * require much less user boiler-plate code, for example no need for multiple steps to create and manage KSP, SNES, etc solver objects
#  * use optional arguments for controlling much of the solution process; requiring less use of setXXXX()
#  * leverage PETSc and petsc4py API as much as possible, avoid reproducing duplicate functionality or APIs
#  * allow direct interactions between PETSc objects and numpy objects in new API calls
#    (for example numpy arrays handled as Vec and Mat automatically)
#  * be "compatible" with other Python solver libraries like Scipy
#  * minimize the amount of new code that must be written
#
#  This is a prototype of the current PETSc functionality. It is not intended to include batching and other new features
#
# Solver Classes:
#      - KSP  -> LinearSystem
#      - SNES -> NonlinearSystem
#      - TS   -> ODE
#      - Tao  -> Optimization
#
# Grid Classes:
#      - DMDA -> Structured
#      - ...
#
#  petsc.xxx exposes the new PETSc solver API
#  petsc.PETSc exposes the petsc4py API
#
#  TODO: Need simple example where user provided class functions are in C or Fortran
#
import sys
try:
  import petsc4py
except ModuleNotFoundError:
  import os
  try:
    petsc4pypath = os.path.join(os.getenv('PETSC_DIR'),os.getenv('PETSC_ARCH'),'lib')
    sys.path.insert(0,petsc4pypath)
    import petsc4py
  except:
    # TODO: I would like to completely avoid the ugly traceback here but not sure how to do it with exiting python
    raise RuntimeError('Unable to locate petsc4py.\nIt is not in your PYTHONPATH or PETSC_DIR and PETSC_ARCH are not set.') from None

petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import re

COMM_WORLD = PETSc.COMM_WORLD
COMM_SELF  = PETSc.COMM_SELF

def convertFromPetscTerminology(str):
  '''Converts petsc4py solver object names, ksp, etc to common names, ls, etc
  '''
  if not str: return ''
  for i,j in {'ksp' : 'ls' ,'snes': 'nls','ts' : 'ode','tao' : 'opt'}.items():
    str = re.sub('_'+i+'_','_'+j+'_',re.sub('-'+i+'_',j+'_',str))
    str = re.sub('\([A-Za-z0-9 ]*\)','',str)
  return str

def convertToPetscTerminology(str):
  '''Converts from common names to petsc4py solver object names, ls, etc to ksp etc
  '''
  for i,j in {'ls' : 'ksp','nls':'snes','ode':'ts','opt':'tao'}.items():
    str = re.sub('_'+i+'_','_'+j+'_',re.sub('^'+i+'_',j+'_',str))
  return str

def convertToVec(v,comm = PETSc.COMM_WORLD,**kwargs):
  '''Converts a numpy array to a PETSc Vec or immediately returns a PETSc Vec
  '''
  if v is None: return v
  if isinstance(v,PETSc.Vec):return v
  return PETSc.Vec().createWithArray(v,comm = comm)
  # TODO: have it tag the numpy array with its Vec so the Vec may be reused the next time

def convertToMat(m,comm = PETSc.COMM_WORLD,**kwargs):
  '''Converts a numpy array to a PETSc Mat or immediately returns a PETSc Mat
  '''
  if m is None: return m
  if isinstance(m,PETSc.Mat):return m
  return PETSc.Mat().createDense(m.shape, array=m, comm=comm)
  # TODO: handle sparse Scipy arrays
  # TODO: have it tag the numpy array with its Mat so the Mat may be reused the next time

# if we delay everything in the init() to setup() we could inspect the object to make better decisions, for example is there a Grid/DM?
# what functions have been provide in the constructor and in the class definition?

# ***********************************
class Problem:
  '''
     Arguments to constructor and all methods:
       kwargs - options for controlling the solver, use help(object) to display them

     Vector and Matrix arguments may be either numpy arrays or petsc.PETSc.Vec or petsc.PETSc.Mat

     Arguments to the constructor and/or class methods:
       solution - vector to hold the problem's solution
       createLeftVector() - a routine that creates the vectors used to contain function or gradient evaluations etc
       createRightVector() - a routine that creates vectors to contain the solution etc
  '''
  def __init__(self, comm = PETSc.COMM_WORLD, type = 'default', solution = None, createLeftVec = None, createRightVec = None,**kwargs):
    import inspect
    if not isinstance(super(),object): super().__init__(comm = comm,**kwargs) # WTF needed for cooperative inheritance with petsc.Grid
    self.comm = comm
    if not solution is None: self.solution = convertToVec(solution)
    if createLeftVec: self.createLeftVec = createLeftVec
    if createRightVec: self.createRightVec = createRightVec
    self.Solver.create(self.comm)

  # cannot be merged with __init__() because it needs to be called after TSSetIFunction() which must be called after the __init__()
  def initialSetOptionsKwargs(self,**kwargs):
    ''' 1) sets the options for the solver from both the default options database and kwargs
        2) saves into the object's help string the kwargs supported by the object, that can be displayed with help(object)
    '''
    PETSc.petscHelpPrintfStringBegin(self.comm)
    self.Solver.setFromOptions()
    self.setOptionsFromKwargs(**kwargs)
    self.__doc__ = self.__doc__ + convertFromPetscTerminology(PETSc.petscHelpPrintfStringEnd(self.comm))

  def setOptionsFromKwargs(self,**kwargs):
    '''Calls self.Solver.setFromOptions() with the given kwargs arguments
       For example: setOptionsFromKwargs(ls,ls_type='gmres',ls_monitor="")
       The keywoard options='a list of PETSc options database entries' is handled as a special case
       For example: setOptionsFromKwargs(ls,ls_type='gmres',petsc_options="-ksp_monitor -ksp_view")
    '''
    # Requiring the empty string for example with ls_monitor='' is annoying, would be nice not to need it
    options = ''
    for k,v in kwargs.items():
      if k == 'petsc_options':
        options = options + ' ' + v + ' '
      k = convertToPetscTerminology(k)
      options = options + '-' + k + ' ' + str(v) + ' '
    opts = PETSc.Options()
    opts.create()
    opts.insertString(options)
    opts.push()
    self.Solver.setFromOptions()
    opts.pop()

  def createLeftVector(self,**kwargs):
    '''Sometimes overloaded by subclasses
       Uses information available in self if not overloaded
    '''
    if hasattr(self,'matrix'):
      (vector,f) = self.matrix.createVecs()
      return convertToVec(vector,**kwargs)
    elif hasattr(self,'rhs'):
      return self.rhs.duplicate()
    else: # the next line is wrong, assumes input and output are same size
      return self.createRightVector(**kwargs)

  def createRightVector(self,**kwargs):
    '''Sometimes overloaded by subclasses
       Uses information available in self if not overloaded
    '''
    if hasattr(self,'matrix'):
      (f,vector) = self.matrix.createVecs()
    elif hasattr(self,'solution'):
      vector = self.solution.duplicate()
    # TODO: handle error
    return convertToVec(vector,**kwargs)

# ***********************************
class LinearSystem(Problem):
  '''
       solution - vector to hold solution, may contain initial guess
       rhs - vector containing or to contain the right hand side
       evalRHS() -  evaluates the right hand side

       matrix - linear operator
       createMatrix() - creates the matrix (does not fill its numerical values)
       evalMatrix() - evaluates the values in the matrix
  '''
  def __init__(self, comm = PETSc.COMM_WORLD,rhs = None, evalRHS = None, matrix=None, createMatrix = None, evalMatrix = None, **kwargs):
    self.__doc__ = Problem.__doc__ + LinearSystem.__doc__
    if not rhs is None: self.rhs = convertToVec(rhs)

    if createMatrix: self.createMatrix = createMatrix
    if not matrix is None: self.matrix = convertToMat(matrix)
    if not hasattr(self,'matrix'):
      self.matrix = convertToMat(self.createMatrix(**kwargs))

    self.Solver = self.KSP = PETSc.KSP(comm)
    super().__init__(comm = comm,**kwargs)
    self.KSP.setOperators(self.matrix)
    self.initialSetOptionsKwargs(**kwargs)

  def solve(self,rhs=None,solution=None,**kwargs):
    self.setOptionsFromKwargs(**kwargs)
    if rhs is None:
      if not hasattr(self,'rhs'): self.rhs = convertToVec(self.createLeftVector(**kwargs))
    else:
      self.rhs = convertToVec(rhs)
    if hasattr(self,'evalRHS'):
      self.evalRHS(self.rhs)
      self.rhs.assemble()
    if solution is None:
      if not hasattr(self,'solution'): self.solution = convertToVec(self.createRightVector(**kwargs))
    if hasattr(self,'evalInitialGuess'): self.evalInitialGuess(self.KSP,self.solution)
    if hasattr(self,'evalMatrix'):  self.evalMatrix(self.matrix)
    self.KSP.solve(self.rhs, self.solution)
    return self.solution

# ***********************************
class NonlinearSystem(Problem):
  '''
       solution - vector to hold solution, may contain initial guess
       evalInitialGuess() - evaluates the initial guess

       rhs - vector to contain (or containing) the right hand side
       evalRHS() - a routine that evaluates the right hand side

       evalFunction() - a routine that evaluates the nonlinear function to be solved with

       jacobian - matrix to store Jacobian values
       createJacobian() - a routine that creates the matrix (does not fill its numerical values)
       evalJacobian() - a routine that evaluates the values in the matrix
  '''
  def __init__(self, comm = PETSc.COMM_WORLD, rhs = None, evalRHS = None, evalInitialGuess = None, evalFunction = None,
               jacobian=None, createJacobian = None, evalJacobian = None,**kwargs):
    self.__doc__ = Problem.__doc__ + NonlinearSystem.__doc__
    if rhs: self.rhs = convertToVec(rhs)
    self.Solver = self.SNES = PETSc.SNES(comm=comm)
    super().__init__(comm = comm, **kwargs)

    if evalRHS: self.evalRHS = evalRHS

    if not jacobian is None: self.jacobian = self.matrix = convertToMat(jacobian)
    if createJacobian: self.createJacobian = createJacobian
    if evalJacobian: self.evalJacobian = evalJacobian
    if hasattr(self,'evalJacobian'):
      if not hasattr(self,'jacobian'): self.jacobian = self.matrix = convertToMat(self.createJacobian(**kwargs))
      self.SNES.setJacobian(self.ievalJacobian,self.jacobian)

    f = self.createLeftVector(**kwargs)
    if evalFunction: self.evalFunction = evalFunction
    self.SNES.setFunction(self.ievalFunction,f)

    f = self.createRightVector(**kwargs)
    if evalInitialGuess: self.evalInitialGuess = evalInitialGuess
    # if hasattr(self,'evalInitialGuess'): self.SNES.setComputeInitialGuess(self.evalInitialGuess,f)
    self.initialSetOptionsKwargs(**kwargs)

  def solve(self,solution=None,rhs=None,**kwargs):
    self.setOptionsFromKwargs(**kwargs)
    if not solution is None: self.solution = convertToVec(solution)
    else:
      if not hasattr(self,'solution'): self.solution = convertToVec(self.createRightVector(**kwargs))
    # maybe need to handle initial guess here
    if rhs is None and hasattr(self,'rhs'): rhs = self.rhs # need to handle evalRHS()
    self.SNES.solve(rhs, self.solution)
    return self.solution

  # stub methods for SNES that remove the SNES argument
  def ievalFunction(self, snes,x, f):
    self.evalFunction(x,f)
    f.assemble()

  def ievalJacobian(self, snes, x, J, P):
    P.zeroEntries()
    import inspect
    args = inspect.getfullargspec(self.evalJacobian).args
    if (args[0] == 'self' and len(args) == 3) or (len(args) == 2):
      self.evalJacobian(x,P)
      if (not J == P): J.assemble()
    else:
      self.evalJacobian(x,J,P)
    P.assemble()

# using this solver does not require explicitly creating PETSc objects
def solve(solution = None, rhs = None,evalFunction=None, **kwargs):
  if not evalFunction is None:
    nls = NonlinearSystem(evalFunction = evalFunction,**kwargs)
    return nls.solve(solution = solution,rhs=rhs)
  else:
    ls = LinearSystem(**kwargs)
    return ls.solve(rhs = rhs,solution=solution)

# ***********************************
class ODE(Problem):
  '''
       solution - vector to contain solution (may contain initial conditions)
       evalInitialConditions() - evaluates the initial conditions of the ODE

       evalRHSFunction() -  evaluates the nonlinear function that defines the right hand side of an explicitly defined ODE

       rhsjacobian - empty Jacobian of right hand side of ODE
       createRHSJacobian() - creates an empty Jacobian of right hand side of ODE
       evalIRHSJacobian() -  evaluates the values in the Jacobian of evalRHSFunction()

       evalIFunction() -  nonlinear function that defines an implicit ODE

       ijacobian - empty Jacobian of left hand side of ODE
       createIJacobian() - creates an empty Jacobian of left hand side of ODE
       evalIJacobian() -  evaluates the values in the Jacobian of evalIFunction()

       Ncost - number of cost functions
       RHSJacobianP - matrix to hold Jacobian XXXX values
       createRHSJacobianP() - creates the Jacobian XXXXX
       evalRHSJacobianP() - evalulates the Jacobian XXXX

       Nparameters - number of paramaters for the ODE
  '''
  def __init__(self, comm = PETSc.COMM_WORLD,  evalRHSFunction = None, rhsjacobian = None, createRHSJacobian = None,
               evalRHSJacobian = None, evalInitialConditions=None, evalIFunction = None, ijacobian = None, createIJacobian = None,
               evalIJacobian = None, Ncost = 1, rhsjacobianp = None, createRHSJacobianP = None, evalRHSJacobianP = None,
               Nparameters = 0, adj_p = None, **kwargs):
    self.__doc__ = Problem.__doc__ + ODE.__doc__
    self.Solver = self.TS = PETSc.TS(comm = comm)
    super().__init__(comm = comm,**kwargs)
    if evalInitialConditions: self.evalInitialConditions = evalInitialConditions

    if not rhsjacobian is None: self.rhsjacobian = self.matrix = convertToMat(rhsjacobian)
    if createRHSJacobian: self.createRHSJacobian = createRHSJacobian
    if evalRHSJacobian: self.evalRHSJacobian = evalRHSJacobian
    if hasattr(self,'evalRHSJacobian'):
      if not hasattr(self,'rhsjacobian'): self.rhsjacobian = self.matrix = convertToMat(self.createRHSJacobian(**kwargs))
      self.TS.setRHSJacobian(self.ievalRHSJacobian,self.rhsjacobian)

    if evalRHSFunction: self.evalRHSFunction = evalRHSFunction
    if hasattr(self,'evalRHSFunction'):
      f = self.createRightVector(**kwargs)
      self.TS.setRHSFunction(self.ievalRHSFunction,f)

    if not ijacobian is None: self.ijacobian = self.matrix = convertToMat(ijacobian)
    if createIJacobian: self.createIJacobian = createIJacobian
    if evalIJacobian: self.evalIJacobian = evalIJacobian
    if hasattr(self,'evalIJacobian'):
      if not hasattr(self,'ijacobian'): self.ijacobian = self.matrix = convertToMat(self.createIJacobian(**kwargs))
      self.TS.setIJacobian(self.ievalIJacobian,self.ijacobian)

    if evalIFunction: self.evalIFunction = evalIFunction
    if hasattr(self,'evalIFunction'):
      f = self.createRightVector(**kwargs)
      self.TS.setIFunction(self.ievalIFunction,f)

    self.Ncost = Ncost
    if adj_p: self.adj_p = adj_p
    if not rhsjacobianp is None: self.rhsjacobianp = convertToMat(rhsjacobianp)
    if createRHSJacobianP: self.createRHSJacobian = createRHSJacobianP
    if evalRHSJacobianP: self.evalRHSJacobian = evalRHSJacobianP
    if hasattr(self,'evalRHSJacobianP'):
      if not hasattr(self,'rhsjacobianp'):
        if hasattr(self,'createRHSJacobianP'): self.rhsjacobianp = convertToMat(self.createRHSJacobianP())
        else: raise RuntimeError("You did not provide a rhsjacobianp")
      self.TS.setRHSJacobianP(self.ievalRHSJacobianP, self.rhsjacobianp)

    self.initialSetOptionsKwargs(**kwargs)

  def integrate(self,solution=None,**kwargs):
    self.setOptionsFromKwargs(**kwargs)
    if not solution is None: self.solution = convertToVec(solution)
    if not hasattr(self,'solution'): self.solution = convertToVec(self.createRightVector(**kwargs))
    if hasattr(self,'evalInitialConditions'): self.evalInitialConditions(self.solution)

    self.TS.solve(self.solution)
    if not hasattr(self,'rhsjacobianp'): return self.solution
    if not hasattr(self,'adj_u'):
      self.adj_u = []
      for i in range(0,self.Ncost):
        self.adj_u.append(convertToVec(self.createRightVector(**kwargs)))

    self.adj_u[0][0] = 1
    self.adj_u[0][1] = 0
    self.adj_u[0].assemble()
    self.adj_u[1][0] = 0
    self.adj_u[1][1] = 1
    self.adj_u[1].assemble()

    self.adj_p[0][0] = 0
    self.adj_p[0].assemble()
    self.adj_p[1][0] = 0
    self.adj_p[1].assemble()

    self.TS.setCostGradients(self.adj_u,self.adj_p)
    self.TS.adjointSolve()
    return self.solution,self.adj_u,self.adj_p

  def ievalInitialConditions(self,ts,u):
    self.evalInitialConditions(u)

  # stub methods for TS that remove the TS argument
  def ievalIFunction(self, ts,t,u,du,F):
    self.evalIFunction(t,u,du,F)

  def ievalIJacobian(self,ts,t,u,du,a,J,P):
    P.zeroEntries()
    import inspect
    args = inspect.getfullargspec(self.evalIJacobian).args
    if (args[0] == 'self' and len(args) == 6) or (len(args) == 5):
      self.evalIJacobian(t,u,du,a,P)
      if (not J == P): J.assemble()
    else:
      self.evalIJacobian(t,u,du,a,J,P)
    P.assemble()

  def ievalRHSFunction(self, ts,t,u,F):
    self.evalRHSFunction(t,u,F)

  def ievalRHSJacobian(self,ts,t,u,J,P):
    P.zeroEntries()
    import inspect
    args = inspect.getfullargspec(self.evalRHSJacobian).args
    if (args[0] == 'self' and len(args) == 5) or (len(args) == 4):
      self.evalRHSJacobian(t,u,P)
      if (not J == P): J.assemble()
    else:
      self.evalRHSJacobian(t,u,J,P)
    P.assemble()

  def ievalRHSJacobianP(self,ts, t, u, Jp):
    self.evalRHSJacobianP(t, u, Jp)
    Jp.assemble()

# using this integrator does not require explicitly creating PETSc objects
def integrate(solution=None,**kwargs):
  ode = ODE(solution=solution,**kwargs)
  return ode.integrate(solution = solution)

# ***********************************
class Optimization(Problem):
  '''
       solution - vector to hold solution, may contain initial guess
       evalInitialGuess() - evaluates the initial guess for the optimization problem

       evalObjective() - evaluates the objective function that is to be optimized
       evalGradient() - evaluates the gradient of evalObjective()

       hessian - empty matrix to hold Hessian values
       createHessian() - creates an empty matrix to hold Hessian
       evalHessian() - evaluates the Hessian of the evalObjective()
  '''
  def __init__(self, solution = None,comm = PETSc.COMM_WORLD,evalObjective = None, evalGradient = None, hessian = None,
               evalHessian = None, evalInitialGuess = None, **kwargs):
    self.__doc__ = Problem.__doc__ + Optimization.__doc__
    self.Solver = self.Tao = PETSc.TAO(comm = comm)
    super().__init__(comm = comm,**kwargs)

    if not solution is None: self.solution = convertToVec(solution)
    if evalInitialGuess: self.evalInitialGuess = evalInitialGuess

    if evalObjective: self.evalObjective = evalObjective
    self.Tao.setObjective(self.ievalObjective)

    if evalGradient: self.evalGradient = evalGradient
    if hasattr(self,'evalGradient'):
      f = self.createLeftVector()
      self.Tao.setGradient(self.ievalGradient,f)

    if evalHessian: self.evalHessian = evalHessian
    if hasattr(self,'evalHessian'):
      self.Tao.setHessian(self.ievalHessian)

    self.initialSetOptionsKwargs(**kwargs)

  def optimize(self,solution = None,**kwargs):
    self.setOptionsFromKwargs(**kwargs)
    if solution is None:
      if not hasattr(self,'solution'): self.solution = convertToVec(self.createRightVector(**kwargs))
    # need to evaluate initial guess
    self.Tao.solve(self.solution)
    return self.solution

  # stub methods for Tao that remove the Tao argument
  def ievalInitialGuess(self,tao, x):
    self.evalInitialGuess(x)

  def ievalObjective(self,tao, x):
    return self.evalObjective(x)

  def ievalGradient(self, tao, x, g):
    self.evalGradient(x, g)
    g.assemble()

  def ievalHessian(self): # TODO: add arguments
    pass

  def iEqualityConstraints(self): #TODO: add arguments
    '''Overloaded by user.
    '''
    pass

# using this optimizer does not require explicitly creating an PETSc objects
def optimize(solution = None,**kwargs):
  opt = Optimization(solution=solution,**kwargs)
  return opt.optimize(solution=solution)

# ***********************************
# ***********************************
class Grid():
  '''The base object for all PETSc grid and discretization types
     DM provides the matrices and vectors and possibly the adapts the function calls
  '''
  def __init__(self, comm = PETSc.COMM_WORLD, **kwargs):
    super().__init__(comm = comm,**kwargs)  # WTF needed for cooperative inheritance
    self.comm = comm

  # these overload the base operations provided in Problem
  def createLeftVector(self,**kwargs):
    return self.DM.createGlobalVector()

  def createRightVector(self,**kwargs):
    return self.DM.createGlobalVector()

# ***********************************
class Structured(Grid):
  '''Structured grid object'''
  def __init__(self, dimensions=(2,2), comm = PETSc.COMM_WORLD, **kwargs):
    # TODO: add support for passing the evalXXXLocal() functions here
    self.DM = self.da = PETSc.DMDA().create(dimensions,stencil_width = 1, comm = comm, **kwargs)
    super().__init__(comm=comm,**kwargs)

  def createMatrix(self,**kwargs):
    return self.DM.createMat()

  def ievalInitialGuess(self, snes, x):
    '''Dispatches to DMDA local function
    '''
    # if would be nice if we didn't need the Local visible to users; but this might require NonlinearSystem to know if DM exists when calling
    # setFunction() and friends see around line 186
    self.evalInitialGuessLocal(self.DM.getVecArray(x))

  def ievalRHS(self,x):
    '''Dispatches to DMDA local function
    '''
    self.evalRHSLocal(self,self.da.getVecArray(x))

  def ievalFunction(self, snes, x, f):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:  # actually need a smarter test for case with nontrivial boundary conditions that require local
      with self.da.globalToLocal(x) as xlocal:
        self.evalFunctionLocal(self.da.getVecArray(xlocal,readonly=1),self.da.getVecArray(f))
    else:
      self.evalFunctionLocal(self.da.getVecArray(x,readonly=1),self.da.getVecArray(f))

  def ievalJacobian(self, snes, x, J, P):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(x) as xlocal:
        self.evalJacobianLocal(self.da.getVecArray(xlocal,readonly=1),P)  # inspect the user evalJacobianLocal to see if it takes both J and P?
    else:
      self.evalJacobianLocal(self.da.getVecArray(x,readonly=1),P)
    if not P == J: J.assemble()

  # the problem is that the DM needs to know about all the PETSc Problem types, KSP, SNES, ODE, ...
  # this would be true with PETSc C API if we actively support the Local versions for all solvers

  def ievalInitialConditions(self,u):
    '''Dispatches to DMDA local function
    '''
    self.evalInitialConditionsLocal(self,self.da.getVecArray(u))

  def ievalIFunction(self, ts,t,u,du,F):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocalVector(u),self.da.globalToLocal(du) as ulocal,dulocal:
        self.evalIFunctionLocal(self, t,self.da.getVecArray(ulocal,readonly=1),self.da.getVecArray(dulocal,readonly=1),self.da.getVecArray(F))
    else:
      self.evalIFunctionLocal(self, t,self.da.getVecArray(u,readonly=1),self.da.getVecArray(du,readonly=1),self.da.getVecArray(F))

  def ievalIJacobian(self,ts,t,u,du,a,J,P):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
     with self.da.globalToLocal(u),self.da.globalToLocal(du) as ulocal,dulocal:
        self.evalIJacobianLocal(self, t,self.da.getVecArray(ulocal,readonly=1),self.da.getVecArray(dulocal,readonly=1),P)
    else:
      self.evalIJacobianLocal(self, t,self.da.getVecArray(u,readonly=1),self.da.getVecArray(du,readonly=1),P)
    if not P == J: J.assemble()

  def ievalRHSFunction(self, ts,t,u,F):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalRHSFunctionLocal(self, t,self.da.getVecArray(ulocal,readonly=1),self.da.getVecArray(F))
    else:
      self.evalRHSFunctionLocal(self, t,self.da.getVecArray(u,readonly=1),self.da.getVecArray(F))

  def ievalRHSJacobian(self,ts,t,u,J,P):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalRHSJacobianLocal(self, t,self.da.getVecArray(ulocal,readonly=1),P)
    else:
      self.evalRHSJacobianLocal(self, t,self.da.getVecArray(u,readonly=1),P)
    if not P == J: J.assemble()

  def ievalObjective(self,tao, u):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalObjectiveLocal(self, self.da.getVecArray(ulocal))
    else:
      self.evalObjectiveLocal(self, self.da.getVecArray(u))

  def ievalGradiant(self, tao, u, g):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalGradiantLocal(self,  self.da.getVecArray(ulocal,readonly=1), self.da.getVecArray(g))
    else:
      self.evalGradiantLocal(self,  self.da.getVecArray(u,readonly=1), self.da.getVecArray(g))

  def ievalHessian(self): # TODO: add arguments
    '''Dispatches to DMDA local function
    '''
    pass

# there could be a subclass of Structured where users only provide function evaluations for inside the inner loops of the
# function and Jacobian evaluation

# ***********************************
class Staggered(Grid):
  '''Staggered structured grid object'''  # could this share Structured code above it ds.getVecArray() worked for staggered grids?
  pass