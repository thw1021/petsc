#!/usr/bin/python3

# Follow instructions at https://www.alcf.anl.gov/support-center/aurorasunspot/getting-started-aurora
# to set up the proxy settings in your .bashrc and git with SSH protocol in your .ssh/config

# module use /soft/modulefiles
# module load spack-pe-oneapi cmake python
#
# Currently Loaded Modules:
#   1) gcc/11.2.0                                      5) libfabric/1.15.2.0     9) spack-pe-oneapi/0.5-rc1
#   2) mpich/51.2/icc-all-pmix-gpu                     6) cray-pals/1.2.12      10) cmake/3.26.4-gcc-testing
#   3) intel_compute_runtime/release/agama-devel-551   7) cray-libpals/1.2.12   11) python/3.10.10-gcc-testing
#   4) oneapi/eng-compiler/2022.12.30.003              8) spack-pe-gcc/0.5-rc1

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpifort',
    '--with-debugging=0',
    '--with-mpiexec-tail=gpu_tile_compact.sh',
    '--SYCLPPFLAGS=-Wno-tautological-constant-compare',
    '--with-sycl',
    '--with-syclc=icpx',
    '--with-sycl-arch=pvc',
    '--COPTFLAGS=-O2 -g',
    '--FOPTFLAGS=-O2 -g',
    '--CXXOPTFLAGS=-O2 -g',
    '--SYCLOPTFLAGS=-O2 -g',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--with-kokkos-kernels-tpl=0' # TODO: wait for KK to fix problems linking libmkl_sycl.so
  ]
  configure.petsc_configure(configure_options)
