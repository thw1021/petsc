import config.package
import os

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.pkgname                = 'firedrake'
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    self.PrefixWriteCheck       = 0

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.slepc           = framework.require('config.packages.slepc',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.petsc4py        = framework.require('config.packages.petsc4py',self)
    self.fftw            = framework.require('config.packages.fftw',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.hdf5            = framework.require('config.packages.hdf5',self)
    self.metis           = framework.require('config.packages.metis',self)
    self.pnetcdf         = framework.require('config.packages.pnetcdf',self)
    self.scalapack       = framework.require('config.packages.scalapack',self)
    self.suitesparse     = framework.require('config.packages.SuiteSparse',self)
    self.zlib            = framework.require('config.packages.zlib',self)
    self.bison           = framework.require('config.packages.bison',self)
    self.ptscotch        = framework.require('config.packages.PTScotch',self)
    self.mumps           = framework.require('config.packages.MUMPS',self)
    self.netcdf          = framework.require('config.packages.netcdf',self)
    self.superlu_dist    = framework.require('config.packages.SuperLU_DIST',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    self.deps            = [self.mpi,self.blasLapack,self.petsc4py,self.slepc,self.fftw,self.hwloc,self.hdf5,self.metis,self.pnetcdf,self.scalapack,self.suitesparse,self.zlib,self.bison,self.ptscotch,self.mumps,self.netcdf,self.superlu_dist,self.hypre]


#need to be passed to pip install also slepc
# PETSC_DIR=/Users/barrysmith/Src/petsc/petsc PETSC_ARCH=arch-firedrake-default HDF5_MPI=ON HDF5_DIR=/opt/homebrew