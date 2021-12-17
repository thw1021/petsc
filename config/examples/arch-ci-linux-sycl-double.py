#!/usr/bin/python

# Kokkos cmake options:
# cmake \
#   -DCMAKE_INSTALL_PREFIX=/nfs/gce/projects/petsc/soft/kokkos \
#   -DCMAKE_CXX_COMPILER=dpcpp \
#   -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
#   -DCMAKE_CXX_STANDARD=17 \
#   -DCMAKE_VERBOSE_MAKEFILE=OFF \
#   -DCMAKE_CXX_EXTENSIONS=OFF \
#   -DCMAKE_BUILD_TYPE=Debug \
#   -DKokkos_ENABLE_SYCL=ON \
#   -DKokkos_ENABLE_SERIAL=ON \
#   -DBUILD_SHARED_LIBS=ON\
#   -DKokkos_ENABLE_DEPRECATED_CODE_3=OFF

# Kokkos-Kernels cmake options:
# cmake \
#   -DCMAKE_CXX_COMPILER=dpcpp \
#   -DBUILD_SHARED_LIBS=ON \
#   -DKokkos_ROOT=/nfs/gce/projects/petsc/soft/kokkos \
#   -DCMAKE_INSTALL_PREFIX=/nfs/gce/projects/petsc/soft/kokkos \
#   -DCMAKE_BUILD_TYPE=Debug

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-mpi-dir=/nfs/gce/projects/petsc/soft/IntelOneAPI/mpi/2021.4.0',
    '--with-fc=0',
    '--COPTFLAGS=-g -O',
    '--CXXOPTFLAGS=-g -O',
    '--CXXPPFLAGS=-std=c++17',
    # use prebuilt Kokkos and KK as it takes a long time to build them from source
    '--with-kokkos-dir=/nfs/gce/projects/petsc/soft/kokkos',
    '--with-kokkos-kernels-dir=/nfs/gce/projects/petsc/soft/kokkos',
    '--with-cuda=0',
    '--with-sycl=1',
    '--with-syclc=dpcpp',
    '--with-sycl-dialect=c++17',
    '--with-sycl-dir=/nfs/gce/projects/petsc/soft/IntelOneAPI/compiler/2021.4.0/linux',
  ]

  configure.petsc_configure(configure_options)
