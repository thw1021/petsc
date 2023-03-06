import petsc
import numpy as np
from numpy import exp, sqrt

class Vanderpol(petsc.ODE):
  n = 2
  def __init__(self, mu_=1.0e3,**kwargs):
    super().__init__(**kwargs)
    self.mu_ = mu_

  def createRHSJacobian(self,**kwargs):
    return petsc.PETSc.Mat().createDense([2,2]).setUp()

  def evalInitialConditions(self,u):
    u[0] = 2.0
    u[1] = -2.0/3.0 + 10.0/(81.0*self.mu_) - 292.0/(2187.0*self.mu_*self.mu_)

  def evalRHSFunction(self, t, u, f):
    f[0] = u[1]
    f[1] = self.mu_*((1.-u[0]*u[0])*u[1]-u[0])

  def evalRHSJacobian(self, t, u, J):
    J[0,0] = 0
    J[1,0] = -self.mu_*(2.0*u[1]*u[0]+1.)
    J[0,1] = 1.0
    J[1,1] = self.mu_*(1.0-u[0]*u[0])

  def createRHSJacobianP(self):
    return petsc.PETSc.Mat().createDense([self.n,1])

  def evalRHSJacobianP(self, t, u, Jp):
    Jp[0,0] = 0
    Jp[1,0] = u[0]-(1.-u[0]*u[0])*u[1]

adj_p = []
adj_p.append(petsc.PETSc.Vec().createSeq(1))
adj_p.append(petsc.PETSc.Vec().createSeq(1))

ode = Vanderpol(ts_problemtype=petsc.PETSc.TS.ProblemType.NONLINEAR,ts_type=petsc.PETSc.TS.Type.RK,Ncost = 2,adj_p = adj_p)

ode.TS.setSaveTrajectory()
ode.TS.setTime(0.0)
ode.TS.setTimeStep(0.0001)
ode.TS.setMaxTime(0.5)
ode.TS.setMaxSteps(1000)
ode.TS.setExactFinalTime(petsc.PETSc.TS.ExactFinalTime.MATCHSTEP)

(u,adj_u,adj_p) = ode.integrate(ts_monitor='')
u.view()
adj_u[0].view()
adj_u[1].view()
adj_p[0].view()
adj_p[1].view()

adj_u = []
adj_u.append(petsc.PETSc.Vec().createSeq(ode.n))
adj_u.append(petsc.PETSc.Vec().createSeq(ode.n))

adj_u[0][0] = 1
adj_u[0][1] = 0
adj_u[0].assemble()
adj_u[1][0] = 0
adj_u[1][1] = 1
adj_u[1].assemble()

adj_p = []
adj_p.append(petsc.PETSc.Vec().createSeq(1))
adj_p.append(petsc.PETSc.Vec().createSeq(1))
adj_p[0][0] = 0
adj_p[0].assemble()
adj_p[1][0] = 0
adj_p[1].assemble()

#ode.TS.setCostGradients(adj_u,adj_p)

#ode.TS.adjointSolve()


