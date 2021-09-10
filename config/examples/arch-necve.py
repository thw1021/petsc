#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  mpilibdir = os.path.join(os.environ['NMPI_ROOT'],'lib64','ve') 
  configure_options = [
    # NEC MPI wrappers explicitly list libmpi.a when linking and not -lmpi
    # checkSharedLinker fails
    '--LDFLAGS=-Wl,-rpath,' + mpilibdir + '-L' + mpilibdir + ' -lmpi',
    '--with-mpiexec=mpiexec -nve 1',
    '--download-sowing-configure-arguments=CC=ncc CXX=nc++',
    '--with-debugging=0',
    '--with-shared-libraries=1',
    'PETSC_ARCH=arch-necve',
  ]
  configure.petsc_configure(configure_options)
