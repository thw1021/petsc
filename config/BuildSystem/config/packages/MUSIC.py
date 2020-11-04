import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://github.com/petsc/MUSIC']
    self.includes          = ['music-c.h']
    self.liblist           = [['libmusic.a','libmusic-c.a']]
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-mpi')
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toString(self.dlib)+'"')
    try:
      output = self.executeShellCommand(self.compilers.CXX + ' -show', log = self.log)[0].split(' ')
      print(output)
      output = ' '.join(output[1:])
      print(output)
    except:
      pass
    # MUSIC configure cannot figure out this flag properly
    args.append('MPI_CXXFLAGS="'+output+'"')
    try:
      output = self.executeShellCommand(self.compilers.CC + ' -show', log = self.log)[0].split(' ')
      print(output)
      output = ' '.join(output[1:])
      print(output)
    except:
      pass
    # MUSIC configure cannot figure out this flag properly
    args.append('MPI_CFLAGS="'+output+'"')
    args.append('MPI_LDFLAGS="'+output+'"')
    return args

  def preInstallDDD(self):
    '''check for configure script - and run bootstrap - if needed'''
    import os
    if not os.path.isfile(os.path.join(self.packageDir,'configure')):
      if not self.programs.libtoolize:
        raise RuntimeError('Could not bootstrap MUSIC using autotools: libtoolize not found')
      if not self.programs.autoreconf:
        raise RuntimeError('Could not bootstrap MUSIC using autotools: autoreconf not found')
      self.logPrintBox('Trying to bootstrap MUSIC using autotools; this may take several minutes')
      try:
        self.executeShellCommand('./bootstrap',cwd=self.packageDir,log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Could not bootstrap MUSIC using autotools: maybe autotools (or recent enough autotools) could not be found?\nError: '+str(e))
