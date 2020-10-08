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
    #'--with-mpi-dir=/opt/intel/inteloneapi/mpi/latest',
    '--with-mpi-dir=/home/glci/soft/mpich-3.3.2-intel',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-syclcxx=dpcpp',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--with-precision=double',
    '--with-clanguage=c',
  ]

  configure.petsc_configure(configure_options)
