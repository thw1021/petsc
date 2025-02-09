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
      ##### getInstallDir calls this, and it sets up self.packageDir (source download), self.confDir and self.installDir
      args = self.formGNUConfigureArgs()  # allow package to change self.packageDir
      if self.download and self.argDB['download-'+self.downloadname.lower()+'-configure-arguments']:
         args.append(self.argDB['download-'+self.downloadname.lower()+'-configure-arguments'])
      args = ' '.join(args)
      conffile = os.path.join(self.packageDir,self.package+'.petscconf')
      fd = open(conffile, 'w')
      fd.write(args)
      fd.close()
      ### Use conffile to check whether a reconfigure/rebuild is required
      if not self.installNeeded(conffile):
        return self.installDir

      # Not currently used
      # self.preInstall()

      if self.builddir == 'yes':
        folder = os.path.join(self.packageDir, 'petsc-build')
        if os.path.isdir(folder):
          import shutil
          shutil.rmtree(folder)
        os.mkdir(folder)
        self.packageDir = folder
        dot = '..'
      else:
        dot = '.'

      ### Taken from formGNUConfigureArgs()
      args = []
      args.append('LIBDIR='+self.libDir)
      self.pushLanguage('C')
      if not self.installwithbatch and hasattr(self.setCompilers, 'cross_cc'):
        args.append('CC="'+self.setCompilers.cross_cc+'"')
      else:
        args.append('CC="'+self.getCompiler()+'"')
      args.append('CFLAGS="'+self.updatePackageCFlags(self.getCompilerFlags())+'"')
      self.popLanguage()
      if hasattr(self.compilers, 'FC'):
        self.pushLanguage('FC')
        fc = self.getCompiler()
        if self.fortran.fortranIsF90:
          try:
            output, error, status = self.executeShellCommand(fc+' -v', log = self.log)
            output += error
          except:
            output = ''
          if output.find('IBM') >= 0:
            fc = os.path.join(os.path.dirname(fc), 'xlf')
            self.log.write('Using IBM f90 compiler, switching to xlf for compiling ' + self.PACKAGE + '\n')
        args.append('FFLAGS="'+self.updatePackageFFlags(self.getCompilerFlags())+'"')
        if not self.installwithbatch and hasattr(self.setCompilers,'cross_fc'):
          args.append('FC="'+self.setCompilers.cross_fc+'"')
        else:
          args.append('FC="'+fc+'"')
        self.popLanguage()


      ### Build package
      try:
        self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
        if self.parallelMake: pmake = self.make.make_jnp+' '+self.makerulename+' '
        else: pmake = self.make.make+' '+self.makerulename+' '

        output2,err2,ret2  = config.base.Configure.executeShellCommand(self.make.make+' clean', cwd=self.packageDir, timeout=200, log = self.log)
        output3,err3,ret3  = config.base.Configure.executeShellCommand(pmake+' '+' '.join(args), cwd=self.packageDir, timeout=6000, log = self.log)
        self.logPrintBox('Running make python on '+self.PACKAGE+'; this may take several minutes')
        output4,err4,ret4  = config.base.Configure.executeShellCommand(self.make.make+' python'+' '+' '.join(args), cwd=self.packageDir, timeout=1000, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make; make python on '+self.PACKAGE+': '+str(e))
        raise RuntimeError('Error running make; make python on '+self.PACKAGE)
      self.postInstall(output2+err2+output3+err3+output4+err4, conffile)
      return self.installDir
