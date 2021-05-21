import config.package
import os

class Configure(config.package.CMakePackage):
    def __init__(self, framework):
        config.package.CMakePackage.__init__(self, framework)
        self.version          = '6.2.0'
        self.versionname      = 'PASTIX_MAJOR_VERSION.PASTIX_MEDIUM_VERSION.PASTIX_MINOR_VERSION'
        self.download         = ['https://gitlab.inria.fr/solverstack/pastix//uploads/5c09cf4e0c120c45e27bc20d24e7f521/pastix-6.2.0.tar.gz']
        self.liblist          = [['libpastix.a']]
        self.functions        = ['pastixInit']
        self.includes         = ['pastix.h']
        self.precisions       = ['single','double']
        self.downloaddirnames = ['pastix-6.2.0']
        self.fc               = 1
        self.requirescxx11    = 1
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
        self.deps            = [ self.blasLapack, self.pthread, self.hwloc ]
        self.odeps           = [ self.mpi, self.scotch, self.metis ]
        return

    def formCMakeConfigureArgs(self):
        args = config.package.CMakePackage.formCMakeConfigureArgs(self)

        if not self.blasLapack.has_cheaders and not self.blasLapack.mkl:
          raise RuntimeError('PaStiX requires LAPACKE to work')

        if not self.scotch.found and not self.metis.found :
            raise RuntimeError('PaStiX requires an ordering library : either METIS, SCOTCH or PT-SCOTCH')

        args.append('-DPASTIX_LR_TESTINGS=OFF')

        if self.compilerFlags.debugging:
            args.append('-DCMAKE_BUILD_TYPE=Debug')
        else:
            args.append('-DCMAKE_BUILD_TYPE=Release')

        if self.indexTypes.integerSize == 64:
            args.append("-DPASTIX_INT64=ON")
        else:
            args.append("-DPASTIX_INT64=OFF")

        if self.metis.found :
            args.append("-DPASTIX_WITH_METIS=ON")
        else :
            args.append("-DPASTIX_WITH_METIS=OFF")

        if self.scotch.found :
            args.append("-DSCOTCH_DIR="+self.scotch.installDir)
            args.append("-DPASTIX_WITH_SCOTCH=ON")
        else :
            args.append("-DPASTIX_WITH_SCOTCH=OFF")

        if self.mpi.found:
            args.append("-DPASTIX_WITH_MPI=ON")
        else:
            args.append("-DPASTIX_WITH_MPI=OFF")
        return args
