import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '0.7.1'
    self.versionname      = 'CUDSS_VERSION'
    self.versioninclude   = 'cudss.h'
    self.functions        = ['cudssCreate']
    self.includes         = ['cudss.h']
    self.liblist          = [['libcudss.so'], ['libcudss.a']]
    self.precisions       = ['single', 'double']
    self.buildLanguages   = ['CUDA']
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.CUDA', self)
    self.deps = [self.cuda]
    return

  def getSearchDirectories(self):
    '''Also search the CUDA toolkit directory since cuDSS ships bundled with CUDA 12+'''
    dirs = config.package.Package.getSearchDirectories(self)
    # Derive CUDA toolkit root from the nvcc compiler location
    self.pushLanguage('CUDA')
    try:
      nvcc = self.getCompiler()
      self.popLanguage()
      self.getExecutable(nvcc, getFullPath=1, resultName='systemNvcc', setMakeMacro=0)
      if hasattr(self, 'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
        if cudaDir not in dirs:
          dirs.append(cudaDir)
    except Exception:
      self.popLanguage()
    return dirs
