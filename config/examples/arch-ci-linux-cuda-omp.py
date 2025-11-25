#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-make-test-np=15',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=1',
    '--with-openmp',
    '--with-threadsafety',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-umpire',
    '--download-hypre',
    '--download-hypre-configure-arguments=--enable-unified-memory',
    '--with-strict-petscerrorcode',
    # '--download-mpich', # mpich builds with cuda-12.8, but its libmpi.so does not have RPATH to libcudart.so
    '--download-openmpi',
    #'--with-coverage',
  ]

  configure.petsc_configure(configure_options)
