import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '1.18.0'
    self.versionname      = 'UCP_API_VERSION'
    self.versioninclude   = ['ucp/api/ucp_version.h']
    self.download         = ['https://github.com/openucx/ucx/releases/download/v'+self.version+'/ucx-'+self.version+'.tar.gz']
    self.download_git     = ['git://https://github.com/openucx/ucx.git']
    self.includes         = ['ucp/api/ucp.h']
    self.requireMPI       = 0
    self.enabled_cuda     = 0
    self.enabled_rocm     = 0
    self.enabled_ze       = 0
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.odeps           = [self.cuda, self.hip]
    return

  def versionToStandardForm(self,ver):
    '''Reverse the formula: UCP_API_VERSION = major << 24 | minor << 16'''
    # See https://github.com/openucx/ucx/blob/master/src/ucp/api/ucp_version.h.in#L10
    return ".".join(map(str,[int(ver)>>24, (int(ver)>>16) & 0xFF]))

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--without-go') # we don't need these bindings
    args.append('--without-java')

    if self.cuda.found:
      args.append('--with-cuda='+self.cuda.cudaDir)
    elif self.hip.found:
      args.append('--with-rocm='+self.hip.hipDir)
    #TODO --with-ze=(DIR)
    return args

  def gitPreReqCheck(self):
    return self.programs.autoreconf and self.programs.libtoolize

  def preInstall(self):
    if self.retriever.isDirectoryGitRepo(self.packageDir):
      # no need to bootstrap tarballs
      self.Bootstrap('./autogen.sh')

  def configure(self):
    return config.package.Package.configure(self)

  def configureLibrary(self):
    import os
    config.package.Package.configureLibrary(self)
    self.getExecutable('ucx_info', path = os.path.join(self.directory, 'bin'))
    if hasattr(self,'ucx_info'):
      try:
        (out, err, ret) = Configure.executeShellCommand(self.ucx_info + ' -v | grep "Configured with"',timeout = 60, log = self.log, threads = 1)
      except Exception as e:
        self.log.write('NVIDIA utility deviceQuery failed '+str(e)+'\n')
      else:
        if '--with-cuda' in out:
          self.enabled_cuda = 1
        elif '--with-rocm' in out:
          self.enabled_rocm = 1
        elif '--with-ze' in out:
          self.enabled_ze = 1