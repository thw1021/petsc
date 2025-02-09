import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version           = '0.2'
    self.gitcommit         = 'v'+self.version
    self.download          = ['git://https://github.com/stevendargaville/PFLARE','https://github.com/stevendargaville/PFLARE/archive/'+self.gitcommit+'.tar.gz']
    self.versionname       = 'PFLARE_VERSION_MAJOR.PFLARE_VERSION_MINOR'
    self.versioninclude    = 'pflare_config.h'
    self.functions         = ['PCRegister_PFLARE']
    self.includes          = ['pflare.h']
    self.liblist           = [['libpflare.a']]
    self.downloadonWindows = 1
    return

  def setupHelp(self,help):
    '''Default GNU setupHelp, but p4est debugging option'''
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument(self.PACKAGE,'-with-pflare-debugging=<bool>',nargs.ArgBool(None,0,"Use pflare's (sometimes computationally intensive) debugging"))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.parmetis   = framework.require('config.packages.parmetis',self)
    self.kokkos     = framework.require('config.packages.kokkos',self)
    self.deps       = [self.mpi,self.blasLapack,self.parmetis]
    self.odeps      = [self.kokkos]
    return

  def formGNUConfigureArgs(self):
    from shlex import quote
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.argDB['with-pflare-debugging']:
      args.append('--enable-debug')
    if self.mpi.mpiexecExecutable:
      args.append('PATH='+quote(os.environ['PATH']+':'+os.path.dirname(self.mpi.mpiexecExecutable)))
    args.append('CPPFLAGS='+quote(self.headers.toStringNoDupes(self.dinclude)))
    args.append('LIBS='+quote(self.libraries.toString(self.dlib)))
    return args

  def Install(self):
    config.package.GNUPackage.setupDependencies(self)
    try:
      self.logPrintBox('Running make python on '+self.PACKAGE+'; this may take several minutes')
      if self.parallelMake: pmake = self.make.make_jnp+' '+self.makerulename+' '
      else: pmake = self.make.make+' '+self.makerulename+' '

      output,err,ret  = config.base.Configure.executeShellCommand(pmake+' python', cwd=self.packageDir, timeout=6000, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running make python on '+self.PACKAGE+': '+str(e))
      raise RuntimeError('Error running make python on '+self.PACKAGE)
    return
