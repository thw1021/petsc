import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion       = '2.7' # ncclSend()/ncclRecv() were introduced in NCCL 2.7
    self.versionname      = 'NCCL_MAJOR.NCCL_MINOR.NCCL_PATCH'
    self.versioninclude   = 'nccl.h'
    self.buildLanguages   = ['CUDA'] # nccl.h includes the CUDA headers, so it has to be tested with nvcc
    self.includes         = ['nccl.h']
    self.liblist          = [['libnccl.so'], ['libnccl_static.a']]
    self.precisions       = ['single','double']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.CUDA', self)
    self.deps = [self.cuda]
    return
