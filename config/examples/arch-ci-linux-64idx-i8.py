#!/usr/bin/python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-64-bit-indices',
    'FFLAGS=-Wall -ffree-line-length-0 -Wno-unused-dummy-argument -fdefault-integer-8',
    '--with-mpi-dir=/nfs/gce/projects/petsc/soft/u22.04/mpich-4.0.2',
    '--with-mpi-ftn-module=mpi_f08',
    '--with-strict-petscerrorcode',
    # '--download-kokkos',
    # '--download-kokkos-commit=e3a19e1c5bd4dcc5f59fba216b72967692f5a33e', # develop as of 4/23/2026
    # '--download-kokkos-kernels',
    # '--download-kokkos-kernels-commit=9b5115e75f95a6153e9b0d99f45a7089858b7c6f', # develop as of 4/23/2026
    '--with-coverage',
  ]
  configure.petsc_configure(configure_options)
