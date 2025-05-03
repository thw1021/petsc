import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    import os
    config.package.Package.__init__(self, framework)
    self.version         = 'v0.9.0'
    self.gitcommit       = self.version
    self.download        = ['git://https://github.com/FEniCS/ffcx/']
    self.functions       = []
    self.includes        = []
    self.liblist         = []
    self.buildLanguages  = ['python']
    self.pkgname         = 'ffcx'
    self.useddirectly    = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.python        = framework.require('config.packages.python', self)
    return

  def Install(self):
    # TODO: set CXX, CXXFLAGS environmental variables
    try:
      output,err,ret = config.package.Package.executeShellCommandSeq([[self.python.pyexe, '-m', 'pip', 'install', '--prefix='+self.installDir, self.packageDir]],timeout=30, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running pip install on '+self.packageDir)
    return self.installDir
