import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes             = ['mkl_pardiso.h']
    self.functions            = ['pardisoinit']
    self.liblist              = [[]] # use MKL detected by BlasLapack.py
    self.precisions           = ['single','double']
    self.hastests             = 1
    self.lookforbydefault     = 1
    self.requires32bitintblas = 0
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument(self.PACKAGE,'-with-'+self.package+'=<bool>',nargs.ArgBool(None,self.required+self.lookforbydefault,'Indicate if you wish to test for '+self.name))

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def configureLibrary(self):
    if not self.blasLapack.mkl or (self.blasLapack.has64bitindices and not self.defaultIndexSize == 64):
      self.argDB['with-'+self.package] = 0
      return
    config.package.Package.configureLibrary(self)
    self.usesopenmp = self.blasLapack.usesopenmp
