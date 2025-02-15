#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-clanguage=cxx',
  '--with-debugging=0',

  '--download-mpich=1',
  '--download-mpich-device=ch3:sock',
  '--download-mpich-commit=main',
  '--download-superlu=1',
    '--download-superlu-commit=master',
  '--download-superlu_dist=1',
    '--download-superlu_dist-commit=master',
  '--download-hypre=1',
  '--download-hypre-commit=master',
  '--with-strict-petscerrorcode',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
