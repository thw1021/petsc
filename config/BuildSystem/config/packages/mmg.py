import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '0bf74fa'
    self.versionname      = 'MMG_VERSION_RELEASE'
    self.download         = ['git://https://github.com/MmgTools/mmg.git']
    self.downloaddirnames = ['Mmg']
    self.includes         = ['mmg/libmmg.h']
    self.liblist          = [['libmmg.a','libmmg3d.a']]
    self.functions        = ['MMG5_paramUsage1']
    self.precisions       = ['double']
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.ptscotch      = framework.require('config.packages.PTScotch',self)
    self.deps          = [self.mathlib,self.ptscotch]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if not self.cmake.found:
      raise RuntimeError('CMake is needed to build Mmg')
    args.append('-DUSE_ELAS=OFF')
    args.append('-DUSE_VTK=OFF')
    args.append('-DSCOTCH_DIR:STRING="'+os.path.join(self.installDir)+'"')
    return args

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)

  def Install(self):
    config.package.CMakePackage.Install(self)
    if self.installSudo:
      newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
      newuser = ''
    if self.framework.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      PETSC_DIR  = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      PETSC_ARCH = ''
      prefix     = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
      PETSC_DIR  = self.petscdir.dir
      PETSC_ARCH = self.arch
      prefix     = os.path.join(self.petscdir.dir,self.arch)
    incDir = os.path.join(prefix,'include')
    cpstr = newuser+' cp '+os.path.join(self.packageDir,'petsc-build','src','common','*')+' '+incDir
    # output,err,ret = config.package.Package.executeShellCommand(cpstr,timeout=100,log=self.log)
    return self.installDir
