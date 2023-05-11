import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    super().__init__(framework)
    self.version          = '4.1.1'
    self.gitcommit        = 'v' + self.version
    self.__tarballs       = [
      'https://github.com/pmodels/mpich/releases/download/v'+self.version+'/mpich-'+self.version+'.tar.gz',
       # does not always work from Python? So add in ftp.mcs URL below
      'https://www.mpich.org/static/downloads/'+self.version+'/mpich-'+self.version+'.tar.gz',
      'https://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-'+self.version+'.tar.gz'
    ]
    self.__git_url        = ['git://https://github.com/pmodels/mpich.git']
    self.download         = self.__git_url + self.__tarballs
    self.gitsubmodules    = ['.']
    self.downloaddirnames = ['mpich']
    self.skippackagewithoptions = 1
    self.isMPI = 1
    return

  def setupDependencies(self, framework):
    super().setupDependencies(framework)
    self.compilerFlags   = framework.require('config.compilerFlags',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.python          = framework.require('config.packages.python',self)
    self.odeps           = [self.cuda, self.hip, self.hwloc]
    return

  def setupHelp(self, help):
    super().setupHelp(help)
    import nargs
    help.addArgument('MPICH', '-download-mpich-pm=<hydra, gforker or mpd>',              nargs.Arg(None, 'hydra', 'Launcher for MPI processes'))
    help.addArgument('MPICH', '-download-mpich-device=<ch3:nemesis or see MPICH docs>', nargs.Arg(None, None, 'Communicator for MPI processes'))
    return

  def checkDownload(self):
    if config.setCompilers.Configure.isCygwin(self.log):
      if config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        raise RuntimeError('Cannot download-install MPICH on Windows with cygwin compilers. Suggest installing OpenMPI via cygwin installer')
      else:
        raise RuntimeError('Cannot download-install MPICH on Windows with Microsoft or Intel Compilers. Suggest using MS-MPI or Intel-MPI (do not use MPICH2')
    if self.argDB['download-'+self.downloadname.lower()] and  'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      self.logWrite('Reusing package prefix install of '+self.defaultInstallDir+' for MPICH')
      self.installDir = self.defaultInstallDir
      self.updateCompilers(self.installDir,'mpicc','mpicxx','mpif77','mpif90')
      return self.installDir
    if self.argDB['download-'+self.downloadname.lower()]:
      return self.getInstallDir()
    return ''

  def formGNUConfigureArgs(self):
    '''MPICH has many specific extra configure arguments'''
    args = super().formGNUConfigureArgs()
    args.append('--with-pm='+self.argDB['download-mpich-pm'])
    args.append('--disable-java')
    if self.hwloc.found:
      args.append('--with-hwloc="'+self.hwloc.directory+'"')
      args.append('--with-hwloc-prefix="'+self.hwloc.directory+'"')
    elif 'with-hwloc' in self.framework.clArgDB and not self.argDB['with-hwloc'] :
      args.append('--without-hwloc')
    else:
      args.append('--with-hwloc=embedded')
    # make sure MPICH does not build with optimization for debug version of PETSc, so we can debug through MPICH
    if self.compilerFlags.debugging:
      args.append("--enable-fast=no")
      args.append("--enable-error-messages=all")
      mpich_device = 'ch3:sock'
    else:
      mpich_device = 'ch3:nemesis'
    if self.cuda.found:
      args.append('--with-cuda='+self.cuda.cudaDir)
      if hasattr(self.cuda,'cudaArch'):
        args.append('--with-cuda-sm='+self.cuda.cudaArch) # MPICH's default to --with-cuda-sm=XX is 'all'
      mpich_device = 'ch4:ucx'
    elif self.hip.found:
      args.append('--with-hip='+self.hip.hipDir)
      mpich_device = 'ch4:ofi' # per https://github.com/pmodels/mpich/wiki/Using-MPICH-on-Crusher@OLCF

    if 'download-mpich-device' in self.argDB:
      mpich_device = self.argDB['download-mpich-device']
    args.append('--with-device='+mpich_device)
    # make MPICH behave properly for valgrind
    args.append('--enable-g=meminit')
    if not self.setCompilers.isDarwin(self.log) and config.setCompilers.Configure.isClang(self.setCompilers.CC, self.log):
      args.append('pac_cv_have_float16=no')
    if config.setCompilers.Configure.isDarwin(self.log):
      args.append('--disable-opencl')

    # MPICH configure errors out on certain standard configure arguments
    args = self.rmArgs(args,['--disable-f90','--enable-f90'])
    args = self.rmArgsStartsWith(args,['F90=','F90FLAGS='])
    args.append('PYTHON='+self.python.pyexe)
    args.append('--disable-maintainer-mode')
    args.append('--disable-dependency-tracking')
    return args

  def gitPreReqCheck(self):
    return self.programs.autoreconf and self.programs.libtoolize

  def preInstall(self):
    if self.retriever.isDirectoryGitRepo(self.packageDir):
      # no need to bootstrap tarballs
      self.Bootstrap('./autogen.sh')
    return super().preInstall()

  def Install(self):
    '''After downloading and installing MPICH we need to reset the compilers to use those defined by the MPICH install'''
    if 'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      return self.defaultInstallDir
    installDir = super().Install()
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

  def consistencyChecks(self):
    """
    Check whether user passed --download-mpich and/or --download-mpich-commit

    On success:
    - if the user passes --download-mpich, self.download is set to self.__tarballs
    - if the user passes --download-mpich-commit, self.download is set to self.__git_url

    Throws RuntimeError if:
    - neither --download-mpich or --download-mpich-commit was set, this is a bug in configure!
    """
    # displays the TESTING: consistencyChecks from .... banner
    # for whatever reason this is not done by default?
    self.printTest(self.consistencyChecks)
    package          = self.downloadname.casefold()
    dl_option        = 'download-{}'.format(package)
    dl               = self.argDB.get(dl_option)
    dl_commit_option = 'download-{}-commit'.format(package)
    dl_commit        = self.argDB.get(dl_commit_option)

    if dl_commit:
      dl_val   = self.__git_url
      opt_name = dl_commit_option
      opt_val  = dl_commit
      descr    = 'git url'
    elif dl:
      dl_val   = self.__tarballs
      opt_name = dl_option
      opt_val  = dl
      descr    = 'tarballs'
    elif dl is None and dl_commit is None:
      raise RuntimeError(
        'Neither --{} and --{} was set, yet we are configuring {}? This is a bug in configure.'.format(
          dl_option, dl_commit_option, package.upper()
        )
      )
    else:
      dl_val = None

    if dl_val is None:
      # neither was passed, so we should do nothing
      self.logPrint(
        '{}::consistencyChecks: found --{}={}, and --{}={}, using default:{}'.format(
          self.package.upper(), dl_option, dl, dl_commit_option, dl_commit,
          '\n\t- '.join([''] + self.download)
        )
      )
    else:
      self.download = dl_val
      self.logPrint(
        '{}::consistencyChecks: found --{}={}, using {}(s):{}'.format(
          self.package.upper(), opt_name, opt_val, descr, '\n\t- '.join([''] + self.download)
        )
      )
    return super().consistencyChecks()

  def configure(self):
    return super().configure()
