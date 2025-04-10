import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = 'v5.3.1'
    self.download               = ['git://https://github.com/eschnett/mpitrampoline.git','https://github.com/eschnett/mpitrampoline/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['mpitrampoline']
    self.includes               = []
    self.liblist                = []
    self.precisions             = ['single','double']
    self.functionsFortran       = 0
    self.buildLanguages         = ['C']
    self.minCmakeVersion        = (2,8,3)
    self.isMPIImplementation    = 1
    self.skippackagewithoptions = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.python          = framework.require('config.packages.python',self)
    self.odeps           = [self.cuda, self.hip, self.hwloc]
    return

  def Install(self):
    '''After downloading and installing mpitrampoline we need to reset the compilers to use those defined by the mpitrampoline install'''
    if 'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      return self.defaultInstallDir
    installDir = config.package.CMakePackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpifc','mpifc')
    return installDir
