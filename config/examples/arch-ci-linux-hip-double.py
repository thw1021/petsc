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
    #'--with-mpi-dir=/home/users/balay/soft/instinct/gcc-10.2.0/mpich-4.1',
    #'--with-blaslapack-dir=/home/users/balay/soft/instinct/gcc-10.2.0/fblaslapack',
    '--with-make-np=24',
    '--with-make-test-np=8',
    '--with-hipc=/opt/rocm-6.0.0/bin/hipcc',
    '--with-hip-dir=/opt/rocm-6.0.0',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    'HIPOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-hip=1',
    '--with-precision=double',
    '--with-clanguage=c',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-hypre', # v2.33.0 requires rocm-6.0.0+
    '--download-magma', # magma v2.9.0+ is needed to avoid build failures with rocm-6.0.0
    '--with-magma-fortran-bindings=0',
    '--download-mfem',
    '--download-metis',
    '--with-strict-petscerrorcode',
    #'--with-coverage',
  ]

  configure.petsc_configure(configure_options)
