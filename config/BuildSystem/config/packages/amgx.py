import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = ''
    self.versionname      = ''
    self.download         = ['https://web.cels.anl.gov/projects/petsc/download/externalpackages/amgx-2.4.0.tar.gz']
    self.gitsubmodules    = ['.']
    self.functions        = []
    self.includes         = ['amgx_c.h']
    self.liblist          = [['libamgx.a']]
    self.precisions       = ['double']
    self.cxx              = 1
    self.requires32bitint = 1
    self.maxCxxVersion    = 'c++17' # https://github.com/NVIDIA/AMGX/issues/231
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.CUDA',self)
    self.deps           = [self.mpi,self.cuda]
    return

  def updateControlFiles(self):
    '''Patch AMGx sources for CUDA 12+ where nvToolsExt was removed in favor of nvtx3'''
    import os
    # Patch CMakeLists.txt: replace CUDA::nvToolsExt with CUDA::nvtx3
    cmakefile = os.path.join(self.packageDir, 'CMakeLists.txt')
    if os.path.isfile(cmakefile):
      with open(cmakefile, 'r') as f:
        contents = f.read()
      if 'CUDA::nvToolsExt' in contents:
        self.logPrint('Patching AMGx CMakeLists.txt: replacing CUDA::nvToolsExt with CUDA::nvtx3')
        contents = contents.replace('CUDA::nvToolsExt', 'CUDA::nvtx3')
        with open(cmakefile, 'w') as f:
          f.write(contents)
    # Patch amgx_timer.h: replace #include "nvToolsExt.h" with #include "nvtx3/nvToolsExt.h"
    timerfile = os.path.join(self.packageDir, 'include', 'amgx_timer.h')
    if os.path.isfile(timerfile):
      with open(timerfile, 'r') as f:
        contents = f.read()
      if '#include "nvToolsExt.h"' in contents:
        self.logPrint('Patching AMGx amgx_timer.h: replacing nvToolsExt.h with nvtx3/nvToolsExt.h')
        contents = contents.replace('#include "nvToolsExt.h"', '#include "nvtx3/nvToolsExt.h"')
        with open(timerfile, 'w') as f:
          f.write(contents)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=RelWithTraces')
    #args.append('-DCMAKE_CXX_FLAGS="-O3"')
    #args.append('-DCMAKE_C_FLAGS="-O3"')
    args.extend(self.cuda.getCmakeCUDAArchFlag())
    # AMGx uses its own CUDA_ARCH variable (not CMAKE_CUDA_ARCHITECTURES).
    # Without this, AMGx defaults to building for ALL supported architectures
    # (e.g. 60;70;80;90 for CUDA 12+), which makes the build extremely slow.
    if hasattr(self.cuda, 'cudaArch'):
      args.append('-DCUDA_ARCH=' + self.cuda.cudaArch.replace(',', ';'))
    if not hasattr(self.cuda, 'cudaDir'):
      raise RuntimeError('CUDA directory not detected! Mail configure.log to petsc-maint@mcs.anl.gov.')
    args.append('-DCUDAToolkit_ROOT=' + self.cuda.cudaDir)
    return args
