import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = '1d86d5327a621c030fe61255ce0114116d60d74a' # master 2023-4-2
    self.download               = ['git://https://github.com/Reference-LAPACK/lapack.git','https://github.com/Reference-LAPACK/lapack/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['netlib-lapack']
    self.includes               = []
    self.liblist                = [['libnlapack.a','libnblas.a']]
    self.precisions             = ['single','double']
    self.functionsFortran       = 1
    self.downloadonWindows      = 1
    self.buildLanguages         = ['FC']
    self.minCmakeVersion        = (2,8,3)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DLIBRARY_PREFIX=n')
    # needed for Microsoft Windows compilers since they do not take Unix standard flags
    args.append('-DCMAKE_DEPENDS_USE_COMPILER=FALSE')
    return args

  def Install(self):
    config.package.CMakePackage.Install(self)

    # LAPACK CMake cannot name the generated files with Microsoft compilers with .lib so need to rename them
    if self.framework.getCompiler().find('win') > -1:
      import os
      from shutil import copyfile
      if os.path.isfile(os.path.join(self.installDir,self.libdir,'libnblas.a')):
        copyfile(os.path.join(self.installDir,self.libdir,'libnblas.a'),os.path.join(self.installDir,self.libdir,'libnblas.lib'))
      if os.path.isfile(os.path.join(self.installDir,self.libdir,'libnlapack.a')):
        copyfile(os.path.join(self.installDir,self.libdir,'libnlapack.a'),os.path.join(self.installDir,self.libdir,'libnlapack.lib'))

    return self.installDir
