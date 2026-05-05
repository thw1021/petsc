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
    # AMGX 2.4.0 source is not C++20 compatible (volatile-qualified types, etc.)
    # so we force C++17 internally in formCMakeConfigureArgs rather than capping
    # the global PETSc C++ standard via maxCxxVersion, which would conflict with
    # packages like Kokkos that may require C++20.
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.CUDA',self)
    self.deps           = [self.mpi,self.cuda]
    return

  def updateControlFiles(self):
    # CUDA::nvToolsExt and nvToolsExt.h were moved/renamed in CUDA 12.
    # Patch CMakeLists.txt: replace link target CUDA::nvToolsExt -> CUDA::nvtx3
    cmakeLists = os.path.join(self.packageDir, 'CMakeLists.txt')
    with open(cmakeLists, 'r') as fd:
      content = fd.read()
    patched = content.replace('CUDA::nvToolsExt', 'CUDA::nvtx3')
    if patched != content:
      self.logPrint('AMGX: patching CMakeLists.txt: replacing CUDA::nvToolsExt with CUDA::nvtx3')
      with open(cmakeLists, 'w') as fd:
        fd.write(patched)
    # Patch include/amgx_timer.h: replace #include "nvToolsExt.h" -> #include "nvtx3/nvToolsExt.h"
    timerHeader = os.path.join(self.packageDir, 'include', 'amgx_timer.h')
    if os.path.isfile(timerHeader):
      with open(timerHeader, 'r') as fd:
        content = fd.read()
      patched = content.replace('"nvToolsExt.h"', '"nvtx3/nvToolsExt.h"')
      if patched != content:
        self.logPrint('AMGX: patching include/amgx_timer.h: replacing nvToolsExt.h with nvtx3/nvToolsExt.h')
        with open(timerHeader, 'w') as fd:
          fd.write(patched)
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=RelWithTraces')
    #args.append('-DCMAKE_CXX_FLAGS="-O3"')
    #args.append('-DCMAKE_C_FLAGS="-O3"')
    args.extend(self.cuda.getCmakeCUDAArchFlag())
    if not hasattr(self.cuda, 'cudaDir'):
      raise RuntimeError('CUDA directory not detected! Mail configure.log to petsc-maint@mcs.anl.gov.')
    args.append('-DCUDAToolkit_ROOT=' + self.cuda.cudaDir)
    # AMGX uses its own CUDA_ARCH CMake variable (not CMAKE_CUDA_ARCHITECTURES).
    # Pass a semicolon-separated list of arch numbers, e.g. "80" or "70;80".
    if hasattr(self.cuda, 'cudaArch') and self.cuda.cudaArchIsVersionList():
      args.append('-DCUDA_ARCH=' + ';'.join(self.cuda.cudaArchList()))
    # AMGX 2.4.0 source is not C++20 compatible (volatile-qualified types, etc.
    # see https://github.com/NVIDIA/AMGX/issues/231). Force C++17 for AMGX
    # regardless of the global PETSc C++ standard (which may be C++20 for Kokkos).
    args = self.rmArgsStartsWith(args, '-DCMAKE_CXX_STANDARD=')
    args.append('-DCMAKE_CXX_STANDARD=17')
    return args
