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
    self.functions        = ['ucp_get_version_string']
    self.liblist          = [['libucp.a'],['ucp.lib']]
    self.requireMPI       = 0
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
