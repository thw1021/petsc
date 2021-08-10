import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit       = 'f136d2a3ac89b3203affef1f04da6fde9d60bf7e'
    self.download        = ['git://https://github.com/bldenton/oce.git']
    self.version         = '7.5.0'
    self.minversion      = '7.5.0'
    self.versionname     = 'OCC_VERSION_COMPLETE'
    self.requiresversion = 1
    self.functions       = []
    self.includes        = ['opencascade/Standard_Version.hxx']
    self.liblist         = [['libTKGeomBase.a', 'libTKBrep.a', 'libTKGeomAlgo.a', 'libTKTopAlgo.a', 'libTKPrim.a', 'libTKShHealing.a', 'libTKBO.a', 'libTKBool.a', 'libTKHLR.a', 'libTKFillet.a', 'libTKFeat.a', 'libTKOffset.a', 'libTKMesh.a', 'libTKXMesh.a', 'libTKXSBase.a', 'libTKSTEPBase.a', 'libTKSTEPAttr.a', 'libTKSTEP209.a', 'libTKSTEP.a', 'libTKIGES.a', 'libTKXCAF.a', 'libTKXDEIGES.a',]]
    self.pkgname         = 'opencascade'
    self.cxx             = 1
    self.hastests        = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.deps          = []
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError("%s requires a C++ compiler\n" % self.pkgname)
    return args
