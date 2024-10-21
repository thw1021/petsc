import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit              = 'f38dd83e30136b4e25eb2343813ee4fbd7c16681'
    #self.gitcommit              = '5.2.0'
    self.download               = ['git://https://github.com/flame/libflame.git','https://github.com/flame/libflame/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['libflame']
    self.functions              = ['FLA_Cntl_gemv_obj_create']
    self.includes               = ['FLAME.h']
    self.liblist                = [['libflame.a','libblis.a'],['libflame.a','libblis-mt.a']]
    self.precisions             = ['single','double']
    self.buildLanguages         = ['C','FC']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.blis          = framework.require('config.packages.blis',self)
    self.deps          = [self.blis]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-lapack2flame')
    args.append('--disable-warnings')
    args.append('--enable-max-arg-list-hack')
    if self.argDB['with-shared-libraries']:
      args.append('--enable-dynamic-build')
    return [arg for arg in args if not arg in ['--enable-shared']]
