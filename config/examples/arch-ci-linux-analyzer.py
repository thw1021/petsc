#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=clang',
  '--with-fc=0',
  '--with-cxx=clang++',

  '--download-mpich=1',
  '--download-cmake=1',
  '--download-make=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-hwloc=1',
  '--download-ptscotch=1',
  '--download-superlu_dist=1',
  '--download-hdf5=1',
  '--download-netcdf=1',
  '--download-pnetcdf=1',
  '--download-zlib=1',
  '--download-triangle=1',
  '--download-exodusii=1',
  '--download-ctetgen=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
