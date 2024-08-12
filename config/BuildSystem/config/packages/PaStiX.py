import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '6.4.0'
    self.minversion       = '6.4.0'
    self.gitcommit        = 'v'+self.version
    self.versionname      = 'PASTIX_MAJOR_VERSION.PASTIX_MEDIUM_VERSION.PASTIX_MINOR_VERSION'
    self.download         = ['git://https://gitlab.inria.fr/solverstack/pastix.git',
                             'https://files.inria.fr/pastix/releases/v6/pastix-'+self.version+'.tar.gz',
                             'https://web.cels.anl.gov/projects/petsc/download/externalpackages/pastix-'+self.version+'.tar.gz']
    self.liblist          = [['libpastix.a']]
    self.functions        = ['pastixInit']
    self.includes         = ['pastix.h']
    self.precisions       = ['single','double']
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.indexTypes      = framework.require('PETSc.options.indexTypes', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.metis           = framework.require('config.packages.metis',self)
    self.scotch          = framework.require('config.packages.PTScotch',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.deps            = [self.blasLapack, self.pthread, self.hwloc]
    self.odeps           = [self.mpi, self.scotch, self.metis]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    if not self.libraries.check(self.dlib, 'cblas_dgemm'):
      raise RuntimeError('PaStiX requires a BLAS library with cblas support')

    if not self.libraries.check(self.dlib, 'LAPACKE_dlange'):
      raise RuntimeError('PaStiX requires a LAPACK library with LAPACKE support')

    if not self.scotch.found and not self.metis.found:
      raise RuntimeError('PaStiX requires an ordering library: either METIS or SCOTCH')

    args.append('-DPASTIX_WITH_FORTRAN=OFF')
    args.append('-DPASTIX_LR_TESTINGS=OFF')

    if self.indexTypes.integerSize == 64:
      args.append("-DPASTIX_INT64=ON")
    else:
      args.append("-DPASTIX_INT64=OFF")

    cmake_prefix_path = []
    cmake_include_path = []

    if self.blasLapack.found:
      if 'with-blaslapack-dir' in self.argDB:
        dir = self.argDB['with-blaslapack-dir']
        if os.path.isdir(dir):
          cmake_prefix_path.append(dir)
      if self.blasLapack.include:
        cmake_include_path.append(self.headers.toStringNoDupes(self.blasLapack.include)[2:])
      #if self.blasLapack.dlib:
      #  args.append('-DBLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
      #  args.append('-DLAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    if self.metis.found:
      args.append("-DPASTIX_ORDERING_METIS=ON")
      if 'with-metis-dir' in self.argDB:
        dir = self.argDB['with-metis-dir']
        if os.path.isdir(dir):
          cmake_prefix_path.append(dir)
      if self.metis.include:
        cmake_include_path.append(self.headers.toStringNoDupes(self.metis.include)[2:])
        #args.append('-DMETIS_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.metis.include)[2:]+'"')
      #if self.metis.lib:
      #  args.append('-DMETIS_LIBRARIES="'+self.libraries.toString(self.metis.lib)+'"')
    else :
      args.append("-DPASTIX_ORDERING_METIS=OFF")

    if self.scotch.found:
      args.append("-DPASTIX_ORDERING_SCOTCH=ON")
      if 'with-ptscotch-dir' in self.argDB:
        dir = self.argDB['with-ptscotch-dir']
        if os.path.isdir(dir):
          cmake_prefix_path.append(dir)
      if self.scotch.include:
        cmake_include_path.append(self.headers.toStringNoDupes(self.scotch.include)[2:])
        #args.append('-DSCOTCH_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.scotch.include)[2:]+'"')
      #if self.scotch.lib:
      #  args.append('-DSCOTCH_LIBRARIES="'+self.libraries.toString(self.scotch.lib)+'"')
    else :
      args.append("-DPASTIX_ORDERING_SCOTCH=OFF")

    if self.hwloc.found:
      if 'with-hwloc-dir' in self.argDB:
        dir = self.argDB['with-hwloc-dir']
        if os.path.isdir(dir):
          cmake_prefix_path.append(dir)
      if self.hwloc.include:
        cmake_include_path.append(self.headers.toStringNoDupes(self.hwloc.include)[2:])
        #args.append('-DHWLOC_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.hwloc.include)[2:]+'"')
      #if self.hwloc.lib:
      #  args.append('-DHWLOC_LIBRARIES="'+self.libraries.toString(self.hwloc.lib)+'"')

    if self.mpi.found:
      args.append("-DPASTIX_WITH_MPI=ON")
      if 'with-mpi-dir' in self.argDB:
        dir = self.argDB['with-mpi-dir']
        if os.path.isdir(dir):
          cmake_prefix_path.append(dir)
      if self.mpi.include:
        cmake_include_path.append(self.headers.toStringNoDupes(self.mpi.include)[2:])
    else:
      args.append("-DPASTIX_WITH_MPI=OFF")

    if cmake_include_path:
        cmake_include_path = list(set(cmake_include_path))
        cmake_include_path = ';'.join(cmake_include_path)
        args.append("-DCMAKE_INCLUDE_PATH=\"{0}\"".format(cmake_include_path))

    if cmake_prefix_path:
      cmake_prefix_path = list(set(cmake_prefix_path))
      cmake_prefix_path = ';'.join(cmake_prefix_path)
      args.append("-DCMAKE_PREFIX_PATH=\"{0}\"".format(cmake_prefix_path))

    return args
