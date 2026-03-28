import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '0.7.1'
    self.versionname      = 'CUDSS_VERSION'
    self.versioninclude   = 'cudss.h'
    self.download         = ['https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda12-archive.tar.xz']
    self.download_aarch64 = ['https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-sbsa/libcudss-linux-sbsa-0.7.1.4_cuda12-archive.tar.xz']
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
    '''Also search the CUDA toolkit directory for cuDSS'''
    dirs = config.package.Package.getSearchDirectories(self)
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

  def configure(self):
    '''Select the correct platform-specific download URL before configuring'''
    import platform
    machine = platform.machine()
    if machine == 'aarch64':
      self.download = self.download_aarch64
    elif machine != 'x86_64' and self.argDB.get('download-cudss', 0):
      raise RuntimeError('--download-cudss is not supported on ' + machine
                         + '. Use --with-cudss-dir to point to a manual installation.')
    config.package.Package.configure(self)
    return

  def Install(self):
    import shutil

    conffile = 'petsc.cudss'
    with open(os.path.join(self.packageDir, conffile), 'w') as f:
      f.write(self.version + '\n')

    if not self.installNeeded(conffile):
      return self.installDir

    srcIncDir = os.path.join(self.packageDir, 'include')
    srcLibDir = os.path.join(self.packageDir, 'lib')
    dstIncDir = os.path.join(self.installDir, 'include')
    dstLibDir = os.path.join(self.installDir, 'lib')

    self.logPrintBox('Installing cuDSS; this may take several seconds')
    output = ''

    # Copy include files
    if os.path.isdir(srcIncDir):
      if not os.path.isdir(dstIncDir):
        os.makedirs(dstIncDir)
      for f in os.listdir(srcIncDir):
        shutil.copy2(os.path.join(srcIncDir, f), dstIncDir)

    # Copy library files, preserving symlinks
    if os.path.isdir(srcLibDir):
      if not os.path.isdir(dstLibDir):
        os.makedirs(dstLibDir)
      for f in os.listdir(srcLibDir):
        src = os.path.join(srcLibDir, f)
        dst = os.path.join(dstLibDir, f)
        if os.path.islink(src):
          linkto = os.readlink(src)
          if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
          os.symlink(linkto, dst)
        elif os.path.isfile(src):
          shutil.copy2(src, dstLibDir)

    self.postInstall(output, conffile)
    return self.installDir
