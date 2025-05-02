from __future__ import generators
import config.package
import sysconfig
from pathlib import Path

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '1.87.0'
    self.download          = ['https://archives.boost.io/release/'+self.version+'/source/boost_'+self.version.replace('.','_')+'.tar.bz2',
                              'https://web.cels.anl.gov/projects/petsc/download/externalpackages/boost_'+self.version.replace('.','_')+'.tar.bz2']
    self.includes          = ['boost/multi_index_container.hpp']
    self.liblist           = []
    self.buildLanguages    = ['Cxx']
    self.downloadonWindows = 1
    self.useddirectly      = 0

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('BOOST', '-boost-headers-only=<bool>', nargs.ArgBool(None, 0, 'When true, do not build boost libraries, only install headers'))
    help.addArgument('BOOST', '-boost-libs=<string>',
                    nargs.ArgString(None, '', 'Comma-separated list of Boost binary libraries to build'))

  def Install(self):
    conffile = Path(self.packageDir) / (self.package + '.petscconf')
    conffile.write_text(self.installDir)

    if not self.installNeeded(str(conffile)):
      return self.installDir

    if self.framework.argDB['boost-headers-only']:
      boostIncludeDir = Path(self.installDir) / self.includedir / 'boost'
      self.logPrintBox('Configure option --boost-headers-only is ENABLED ... boost libraries will not be built')
      self.logPrintBox('Installing boost headers, this should not take long')
      try:
        if boostIncludeDir.exists() or boostIncludeDir.is_symlink():
          boostIncludeDir.unlink()
        cmd = f'cd {self.packageDir} && ln -s $PWD/boost {boostIncludeDir}'
        config.base.Configure.executeShellCommand(cmd, timeout=6000, log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Error linking Boost headers:\n'+str(e))
    else:
      if not self.checkCompile('#include <bzlib.h>', ''):
        raise RuntimeError('Boost requires bzlib.h. Please install it in default compiler search location.')

      if not (Path(sysconfig.get_paths()['include']) / 'pyconfig.h').is_file():
        raise RuntimeError('pyconfig.h missing: Boost requires python development version to be installed. (pythonX.x-dev)')

      with self.Language('Cxx'):
          cxx = self.getCompiler().lower()
      if 'mpiicpx' in cxx or 'icpx' in cxx or 'intel' in cxx:
          toolset = 'intel-linux'
      else:
          toolset = 'gcc'

      self.logPrintBox(f'Building Boost with toolset "{toolset}", compiler "{cxx}"')

      jamfile = Path(self.packageDir) / 'user-config.jam'
      jamfile.write_text(f'using {toolset} : : {cxx} : <cxxflags>"-ftemplate-depth=1024" ;\n')
      boost_libs = self.framework.argDB.get('boost-libs','')
      boost_libs_flag = ' '.join(f'--with-{lib.strip()}' for lib in boost_libs.split(',') if lib)
      cmd = (
          f'cd {self.packageDir} && export CXX={cxx} && '
          f'./bootstrap.sh --with-toolset={toolset} --prefix={self.installDir} && '
          f'./b2 toolset={toolset} pch=off cxxstd=20 {boost_libs_flag} -j$(nproc) && '
          f'./b2 toolset={toolset} pch=off cxxstd=20 {boost_libs_flag} -j$(nproc) install'
      )
      out, err, ret = config.base.Configure.executeShellCommand(cmd, timeout=6000, log=self.log)
      self.postInstall(out + err, str(conffile))
    return self.installDir
