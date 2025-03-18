import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'a4e1bb593f369cb8a2f4b846fda09e539b0547b5' # main, jan-19-2025
    #self.gitcommit             = 'v'+self.version
    self.download               = ['git://https://gitlab.com/slepc/slepc.git','https://gitlab.com/slepc/slepc/-/archive/'+self.gitcommit+'/slepc-'+self.gitcommit+'.tar.gz']
    self.functions              = []
    self.includes               = []
    self.skippackagewithoptions = 1
    return

  def Install(self):
    includeDir     = os.path.join(self.installDir, 'include')

    output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
    output,err,ret  = config.package.Package.executeShellCommand('cp -rf '+os.path.join(self.packageDir, 'include/*')+' '+includeDir, timeout=60, log = self.log)
    self.addDefine('HAVE_SLEPC', 1)
    self.addDefine('SLEPC_LIB_DIR','"'+os.path.join(self.installDir,'lib')+'"')  # need addNakeDefine()
    self.addDefine('SLEPC_HAVE_PACKAGES','""')  # need addNakeDefine()    
    self.addDefine('GMAKEGENSRC_ksp', os.path.join(self.packageDir, 'src')) #should be a makeMacro and support multiple packages
    # need to handle extra blas checking in slepc
    # need to handle external packages in slepc
    with open(os.path.join(includeDir,'slepcconf.h'),'w') as f:
      pass
    self.include = [os.path.join(self.packageDir,'include')]   # needed at compile time but should not be in final list provided to user perhaps a new self.cinclude
    self.found = 1
    return self.installDir

  def configureLibrary(self):
    self.checkDownload()
    self.framework.packages.append(self)
    self.addMakeRule('slepc-build','')
    self.addMakeRule('slepc-install','')

  def alternateConfigureLibrary(self):
    self.addMakeRule('slepc-build','')
    self.addMakeRule('slepc-install','')
