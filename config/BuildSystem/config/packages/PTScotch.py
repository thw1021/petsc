import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '7.0.3'
    self.versionname      = 'SCOTCH_VERSION.SCOTCH_RELEASE.SCOTCH_PATCHLEVEL'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://gitlab.inria.fr/scotch/scotch.git',
                             'https://gitlab.inria.fr/scotch/scotch/-/archive/'+self.gitcommit+'/scotch-'+self.gitcommit+'.tar.gz',
                             'https://web.cels.anl.gov/projects/petsc/download/externalpackages/scotch-'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['scotch','petsc-pkg-scotch']
    self.liblist          = [['libptesmumps.a','libptscotchparmetisv3.a','libptscotch.a','libptscotcherr.a','libesmumps.a','libscotch.a','libscotcherr.a'],
                             ['libptesmumps.a','libptscotchparmetis.a','libptscotch.a','libptscotcherr.a','libesmumps.a','libscotch.a','libscotcherr.a'],
                             ['libptesmumps.a','libptscotchparmetis.a','libptscotch.a','libptscotcherr.a','libesmumps.a','libscotch.a','libscotcherr.a']]
    self.functions        = ['SCOTCH_archBuild']
    self.functionsDefine  = ['SCOTCH_ParMETIS_V3_NodeND']
    self.includes         = ['ptscotch.h']
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.deps           = [self.mpi,self.mathlib]
    self.pthread        = framework.require('config.packages.pthread',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.odeps          = [self.pthread,self.zlib]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DINTSIZE:STRING='+ ('64' if self.getDefaultIndexSize() == 64 else '32'))
    args.append('-DINSTALL_METIS_HEADERS:BOOL=OFF')
    args.append('-DSCOTCH_METIS_PREFIX:BOOL=ON')
    if self.setCompilers.isDarwin(self.log):
      #args.append('-DCOMMON_TIMING_OLD')
      args.append('-DCOMMON_OS_MACOS:BOOL=ON')
    if self.setCompilers.isMINGW(self.framework.getCompiler(), self.log):
      args.append('-DCOMMON_OS_WINDOWS:BOOL=ON')
    #-D INSTALL_METIS_HEADERS:BOOL=OFF \
    #-D COMMON_PTHREAD_FILE:BOOL=ON \
    #-D SCOTCH_PTHREAD:BOOL=ON \
    #-D SCOTCH_PTHREAD_MPI:BOOL=ON \
    #-D COMMON_PTHREAD_AFFINITY_LINUX:BOOL=ON
    return args
