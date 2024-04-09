#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=win32fe_cl',
    '--with-cxx=win32fe_cl',
    '--with-fc=0',
    '--with-cudac=win32fe_nvcc',
    '--with-cuda=1',
    #'--with-cuda-arch=80',
    '--with-cuda-include=/cygdrive/c/PROGRA~1/NVIDIA~2/CUDA/v12.3/include',
    '--with-cuda-lib=-L/cygdrive/c/PROGRA~1/NVIDIA~2/CUDA/v12.3/lib/x64 curand.lib cusolver.lib cudart_static.lib cublas.lib cusparse.lib cufft.lib',
    '--with-blaslapack-lib=-L/cygdrive/c/PROGRA~2/Intel/oneAPI/mkl/latest/lib mkl_intel_lp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib',
    '--with-mpi-include=/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest/include',
    '--with-mpi-lib=/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest/lib/impi.lib',
    '--with-mpiexec=/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest/bin/mpiexec -localonly',
  ]
  configure.petsc_configure(configure_options)
