import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = 'v3.12.0'
    self.download               = ['git://https://github.com/Reference-LAPACK/lapack.git','https://github.com/Reference-LAPACK/lapack/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['netlib-lapack','lapack']
    self.includes               = []
    self.liblist                = [['libnlapack.a','libnblas.a']]
    self.precisions             = ['single','double']
    self.functionsFortran       = 1
    self.buildLanguages         = ['FC']
    self.minCmakeVersion        = (2,8,3)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.argDB['download-pastix']:
      args.append('-DCBLAS:BOOL=ON')
      args.append('-DLAPACKE:BOOL=ON')
      self.liblist = [['liblapacke.a','libcblas.a','liblapack.a','libblas.a']]
    else:
      args.append('-DLIBRARY_PREFIX=n')
    return args
