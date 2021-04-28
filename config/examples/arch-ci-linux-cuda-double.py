#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-make-test-np=1',
    '--with-mpi-dir=/home/petsc/soft/openmpi-4.0.2-cuda',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=1',
    '--with-precision=double',
    '--with-clanguage=c',
    '--download-kokkos',
    '--download-kokkos-commit=d4ed86ffb1b156faaa0f936ce839cfd6d3282478',  # on 2021-04-28
    '--download-kokkos-kernels',
    '--download-kokkos-kernels-commit=54e4213ca028163a76639d23b1204b213203bc79', # on 2021-04-28
    '--download-hwloc',
    # Note: If using nvcc with a host compiler other than the CUDA SDK default for your platform (GCC on Linux, clang
    # on Mac OS X, MSVC on Windows), you must set -ccbin appropriately in CUDAFLAGS, as in the example for PGI below:
    # 'CUDAFLAGS=-ccbin pgc++',
  ]

  configure.petsc_configure(configure_options)
