import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '7.7.0'
    self.maxversion       = '7.7.0'
    self.minversion       = '7.7.0'
    self.gitcommit        = 'v7.7.0'
    self.versionname      = 'SUNDIALS_PACKAGE_VERSION'
    self.download         = ['https://github.com/llnl/sundials.git']
    self.downloaddirnames = ['sundials']
    self.functions        = ['CVSpgmr','CVDense']
    self.includes         = ['sundials/sundials_config.h']
    self.parallelMake     = 1
    self.complex          = 0
    self.precisions       = ['single', 'double']
    self.hastests         = 1
    self.useddirectly     = 1
    self.builtafterpetsc  = 1
    self.need35policy     = True

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.scalar        = framework.require('PETSc.options.scalarTypes',self)
    self.hypre        = framework.require('config.packages.hypre',self)
    self.superlu_dist = framework.require('config.packages.hypre',self)
    self.blasLapack   = framework.require('config.packages.BlasLapack',self)
    self.mpi          = framework.require('config.packages.MPI',self)
    self.mathlib      = framework.require('config.packages.mathlib',self)
    self.cuda         = framework.require('config.packages.CUDA',self)
    self.hip          = framework.require('config.packages.HIP',self)
    self.sycl         = framework.require('config.packages.SYCL',self)
    self.openmp       = framework.require('config.packages.OpenMP',self)
    self.odeps        = [self.hypre,self.cuda,self.hip,self.sycl,self.openmp,self.superlu_dist]
    self.deps         = [self.mpi,self.blasLapack,self.mathlib]

  def formCMakeConfigureArgs(self):
    import os
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-B../petsc-build') # needed by funky SUNDIALS CMake sanity checker

    args.append('-DSUNDIALS_ENABLE_PYTHON=OFF')
    args.append('-DSUNDIALS_ENABLE_MPI=ON')
    if self.defaultIndexSize == 64:  args.append('-DSUNDIALS_INDEX_SIZE=64')
    else:  args.append('-DSUNDIALS_INDEX_SIZE=32')
    if self.scalar.precision == 'single': args.append('-DSUNDIALS_PRECISION=single')
    else: args.append('-DSUNDIALS_PRECISION=double')
    args.append('-DBLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DLAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    if self.openmp.found:
      args.append('-DSUNDIALS_ENABLE_OPENMP=ON')
    if self.cuda.found:
      args.append('-DSUNDIALS_ENABLE_CUDA=ON')
    elif self.hip.found:
      args.append('-DSUNDIALS_ENABLE_HIP=ON')
    elif self.sycl.found:
      args.append('-DSUNDIALS_ENABLE_SYCL=ON')

    if self.superlu_dist.found:
      args.append('-DSUNDIALS_ENABLE_SUPERLUDIST=ON')
      args.append('-DSUPERLUDIST_DIR="'+self.superlu_dist.installDir+'"')

    if self.hypre.found:
      args.append('-DSUNDIALS_ENABLE_HYPRE=ON')
      args.append('-DHYPRE_DIR="'+self.hypre.installDir+'"')

    args.append('-DSUNDIALS_ENABLE_C_EXAMPLES=OFF')
    args.append('-DSUNDIALS_ENABLE_CXX_EXAMPLES=OFF')
    args.append('-DSUNDIALS_ENABLE_FORTRAN_EXAMPLES=OFF')
    args.append('-DSUNDIALS_ENABLE_EXAMPLES_INSTALL=OFF')
    args.append('-DSUNDIALS_ENABLE_FORTRAN=OFF')

    args.append('-DSUNDIALS_ENABLE_PETSC=ON')
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      args.append('-DPETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix'])))
      args.append('-DPETSC_ARCH=""')
    else:
      args.append('-DPETSC_DIR='+os.path.join(self.petscdir.dir))
      args.append('-DPETSC_ARCH='+self.arch)

    return args

  def Install(self):
    import os

    output,err,ret  = config.package.Package.executeShellCommand('git describe --abbrev=12 --dirty --always --tags', cwd=self.packageDir)
    if not err and not ret:
      self.foundversion = output

    args = self.formCMakeConfigureArgs()
    if self.download and self.argDB['download-'+self.downloadname.lower()+'-cmake-arguments']:
       args.append(self.argDB['download-'+self.downloadname.lower()+'-cmake-arguments'])
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()

    if not self.installNeeded(conffile):
      return self.installDir
    if not self.cmake.found:
      raise RuntimeError('CMake not found, needed to build '+self.PACKAGE+'. Rerun configure with --download-cmake.')

    # effectively, this is 'make clean'
    folder = os.path.join(self.packageDir, 'petsc-build')
    if os.path.isdir(folder):
      import shutil
      shutil.rmtree(folder)
    os.mkdir(folder)

    # these checks are usually done in configureLibrary
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.directory = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      self.include = '-I'+os.path.join(os.path.abspath(os.path.expanduser(self.argDB['prefix'])),'include')
      self.lib = [os.path.join(os.path.abspath(os.path.expanduser(self.argDB['prefix'])),'lib','libsundials_ida'),'-lsundials_arkode -lsundials_cvode -lsundials_sunnonlinsolpetscsnes -lsundials_nvecpetsc -lsundials_core']
    else:
      self.directory = self.petscdir.dir
      self.include = '-I'+os.path.join(self.petscdir.dir,self.arch,'include')
      self.lib = [os.path.join(self.petscdir.dir,self.arch,'lib','libsundials_ida'),'-lsundials_arkode -lsundials_cvode -lsundials_sunnonlinsolpetscsnes -lsundials_nvecpetsc -lsundials_core']
    self.found     = 1

    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
       carg = 'PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH="" '
       prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
       carg = ''
       prefix = os.path.join(self.petscdir.dir,self.arch)

    # delay the entire process, starting with CMake until after the PETSc libraries are built
    self.addPost(os.path.join(self.packageDir,'petsc-build'),[carg + ' ' + self.cmake.cmake + ' .. ' + args, self.make.make_jnp + '  ' + self.makerulename,'${OMAKE} install'])
    return self.installDir
