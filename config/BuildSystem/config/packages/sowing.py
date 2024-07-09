import config.package
import os

def noCheck(command, status, output, error):
  ''' Do no check result'''
  return

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.minversion        = '1.1.26.12'
    self.gitcommit         = '153edaa35fadca576bf088063dec6d24c9763e04' # add-enum-bfort 'v1.1.26.12'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-sowing.git','https://bitbucket.org/petsc/pkg-sowing/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['petsc-pkg-sowing']
    self.downloadonWindows = 1
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.parallelMake      = 0  # sowing does not support make -j np
    self.executablename    = 'doctext'
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('SOWING', '-download-sowing-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cxx=<prog>',                    nargs.Arg(None, None, 'CXX compiler for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cpp=<prog>',                    nargs.Arg(None, None, 'CPP for sowing configure'))
    help.addArgument('SOWING', '-download-sowing-cxxpp=<prog>',                  nargs.Arg(None, None, 'CXX CPP for sowing configure'))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.petscclone = framework.require('PETSc.options.petscclone', None)
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments since this does not use the MPI compilers etc
       Sowing will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.installDir]
    args.append('CPPFLAGS=-O2')
    if 'download-sowing-cc' in self.argDB and self.argDB['download-sowing-cc']:
      args.append('CC="'+self.argDB['download-sowing-cc']+'"')
    if 'download-sowing-cxx' in self.argDB and self.argDB['download-sowing-cxx']:
      args.append('CXX="'+self.argDB['download-sowing-cxx']+'"')
    if 'download-sowing-cpp' in self.argDB and self.argDB['download-sowing-cpp']:
      args.append('CPP="'+self.argDB['download-sowing-cpp']+'"')
    if 'download-sowing-cxxpp' in self.argDB and self.argDB['download-sowing-cxxpp']:
      args.append('CXXPP="'+self.argDB['download-sowing-cxxpp']+'"')
    return args

  def alternateConfigureLibrary(self):
    '''Check if Sowing download option was selected'''
    self.checkDownload()

  def configure(self):
    if self.framework.batchBodies:
      self.logPrint('In --with-batch mode with outstanding batch tests to be made; hence skipping sowing for this configure')
      return

    if self.petscclone.isClone:
      self.logPrint('PETSc clone, checking for Sowing \n')
      self.getExecutable('pdflatex', getFullPath = 1)

      if 'with-sowing-dir' in self.framework.clArgDB and self.argDB['with-sowing-dir']:
        installDir = os.path.join(self.argDB['with-sowing-dir'],'bin')
      elif self.argDB['download-sowing']:
        #check cygwin has g++
        if os.path.exists('/usr/bin/cygcheck.exe') and not os.path.exists('/usr/bin/g++.exe') and not self.setCompilers.isMINGW(self.framework.getCompiler(), self.log):
          raise RuntimeError("Error! Sowing on Microsoft Windows requires cygwin's g++ compiler. Please install it with cygwin setup.exe and rerun configure")
        config.package.GNUPackage.configure(self)
        installDir = os.path.join(self.installDir,'bin')
      else:
        return

      self.getExecutable('doctext',  path=installDir, getFullPath = 1)
      self.getExecutable('mapnames', path=installDir, getFullPath = 1)
      self.getExecutable('bib2html', path=installDir, getFullPath = 1)
      self.found = 1
      self.foundinpath = 0
