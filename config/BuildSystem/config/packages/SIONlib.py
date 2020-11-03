import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit   = 'master'
    self.versionname = 'SION_MAIN_VERSION.SION_SUB_VERSION.SION_VERSION_PATCHLEVEL'
    self.download    = ['git://https://gitlab.com/petsc/pkg-sionlib']
    self.functions   = ['lsion_parreinit_mpi']
    self.includes    = ['sion_const.h']
    self.liblist     = [['liblsionmpi_64.a','liblsiongen_64.a','liblsionser_64.a','liblsioncom_64.a','liblsioncom_64_lock_none.a','liblsioncom_64_lock_pthreads.a'],\
                        ['liblsionmpi_32.a','liblsiongen_32.a','liblsionser_32.a','liblsioncom_32.a','liblsioncom_32_lock_none.a','liblsioncom_32_lock_pthreads.a']]

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi      = framework.require('config.packages.MPI',self)
    self.deps     = [self.mpi]

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    return args

  def Install(self):
    args = self.formGNUConfigureArgs()
    args = self.rmArgsStartsWith(args,'--libdir')
    args = self.rmArgsStartsWith(args,'--enable-shared')
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(args)
    fd.close()

    if not self.installNeeded(conffile):
      return self.installDir

    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=200, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')

      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+os.path.join(self.packageDir,'build-*')+' && '+self.make.make+' CXXENABLE=1', timeout=600, log = self.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')

      self.installDirProvider.printSudoPasswordMessage(self.installSudo)
      output3,err3,ret3  = config.base.Configure.executeShellCommand('cd '+os.path.join(self.packageDir,'build-*')+' && '+self.installSudo+self.make.make+' install CXXENABLE=1', timeout=30, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make; make install on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3, conffile)
    return self.installDir

