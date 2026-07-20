import config.package
import glob
import os
import platform

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = 'a85ad8827fd0dc9de88bad12d6d95a4022fc02a0'
    self.download          = ['git://https://github.com/libxsmm/libxsmm.git','https://github.com/libxsmm/libxsmm/archive/'+self.gitcommit+'.tar.gz']
    self.includes          = ['libxsmm.h']
    self.liblist           = [['libxsmm.a']]
    self.functions         = ['libxsmm_init']
    self.precisions        = ['single', 'double']
    self.complex           = 0
    self.buildLanguages    = ['C', 'Cxx']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.make = framework.require('config.packages.make', self)
    return

  def Install(self):
    conffile = os.path.join(self.packageDir, self.package + '.petscconf')
    with open(conffile, 'w') as f:
      f.write(self.installDir + '\n')
    if not self.installNeeded(conffile):
      return self.installDir

    args = ['PREFIX=' + self.installDir, 'CC=' + self.getCompiler('C'), 'CXX=' + self.getCompiler('Cxx'), 'STATIC=1', 'WERROR=0']
    if platform.system() == 'Darwin' and platform.machine() == 'arm64': args += ['PLATFORM=1', 'JIT=1']

    try:
      self.logPrintBox('Compiling LIBXSMM; this may take several minutes')
      output, err, ret = config.package.Package.executeShellCommand(self.make.make_jnp_list + args + ['install'], cwd=self.packageDir, timeout=600, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error building/installing LIBXSMM: ' + str(e))

    self.postInstall(output + err, conffile)
    return self.installDir
