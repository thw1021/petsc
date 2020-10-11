import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '2.7.2'
    self.gitcommit        = 'e0a331581e4912a9b0e2f478aedb95bd7154b80d' # barry/fix-swig-lib64 10/10/202
    self.download         = ['git://https://github.com/petsc/ADOL-C.git']
    self.includes         = ['adolc/adolc.h']
    self.liblist          = [['libadolc.a']]
    self.functions        = ['myalloc2','myfree2']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.precisions       = ['double']
    self.complex          = 0
    self.downloaddirnames = ['ADOL-C']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.colpack = framework.require('config.packages.ColPack', self)
    self.deps    = [self.colpack]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--without-boost')
    args.append('--enable-sparse')
    args.append('--with-colpack="'+self.colpack.directory+'"')
    return args
