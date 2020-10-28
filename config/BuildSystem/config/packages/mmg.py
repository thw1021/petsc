import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    # self.version          = '5.5.2'
    self.versionname      = 'MMG_VERSION_RELEASE'
    self.gitcommit        = 'b8f1fe4'
    self.download         = ['git://https://github.com/MmgTools/mmg.git','https://github.com/MmgTools/mmg/archive/'+self.gitcommit+'.tar.gz']
    self.includes         = ['mmg/libmmg.h']
    self.liblist          = [['libmmg.a','libmmg3d.a']]
    self.functions        = ['MMG5_paramUsage1']
    self.precisions       = ['double']
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
    args.append('-DSCOTCH_DIR:STRING="'+self.ptscotch.directory+'"')
    return args

  def configureLibrary(self):
    config.package.CMakePackage.configureLibrary(self)
