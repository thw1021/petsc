import config.package
import glob
import os
import platform

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '2.0.0'
    self.gitcommit         = self.version
    self.download          = ['git://https://github.com/libxsmm/libxsmm.git','https://github.com/libxsmm/libxsmm/archive/'+self.gitcommit+'.tar.gz']
    self.includes          = ['libxsmm.h']
    self.liblist           = [['libxsmm.a']]
    self.functions         = ['libxsmm_init']
    self.precisions        = ['single', 'double']
    self.complex           = 0
    self.versionname       = 'LIBXSMM_VERSION'
    self.buildLanguages    = ['C', 'Cxx']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.make = framework.require('config.packages.make', self)
    return

  def Install(self):
    args     = ['PREFIX=' + self.installDir, 'CC=' + self.getCompiler('C'), 'CXX=' + self.getCompiler('Cxx'), 'STATIC=' + ('0' if self.checkSharedLibrariesEnabled() else '1'), 'PLATFORM=1', 'JIT=1', 'WERROR=0', 'FORTRAN=0']
    conffile = os.path.join(self.packageDir, self.package + '.petscconf')
    with open(conffile, 'w') as f:
      f.write(' '.join(args) + '\n')
    if not self.installNeeded(conffile):
      return self.installDir

    try:
      self.logPrintBox('Compiling LIBXSMM; this may take several minutes')
      output, err, ret = config.package.Package.executeShellCommand(self.make.make_jnp_list + args + ['install'], cwd=self.packageDir, timeout=600, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error building/installing LIBXSMM: ' + str(e))

    self.postInstall(output + err, conffile)
    return self.installDir
