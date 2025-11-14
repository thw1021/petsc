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
    '--with-mpi-dir=/nfs/gce/projects/petsc/soft/u22.04/openmpi-5.0.9/',
    #'--with-blaslapack-dir=/home/users/balay/soft/instinct/gcc-10.2.0/fblaslapack',
    '--with-make-np=24',
    '--with-make-test-np=8',
    '--with-hipc=/opt/rocm-6.2.4/bin/hipcc',
    '--with-hip-dir=/opt/rocm-6.2.4',
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
    '--download-umpire',
    '--download-hypre',
    '--download-magma',
    '--download-mfem',
    '--download-metis',
    '--with-strict-petscerrorcode',
    #'--with-coverage',
  ]

  configure.petsc_configure(configure_options)
