.. _doc_features:

*******************************
Core Features and Functionality
*******************************

PETSc is a general parallel linear and non-linear solver framework, which provides these
general classes of functionality:

.. contents:: Table Of Contents
   :local:
   :backlinks: entry
   :depth: 1


Supported Systems
=================

- :ref:`HPC <doc_config_hpc>`
- :ref:`Linux <doc_config>`
- :ref:`MacOS <doc_config>`
- :ref:`Windows <doc_windows>`

General Features
================

- :ref:`Parallel vectors <chapter_vectors>`
- :ref:`Vector code for communicating ghost points <sec_scatter>`
- :ref:`Parallel matrices <chapter_matrices>`
- Several sparse and dense matrix storage formats

  - AIJ/CSR (Yale sparse matrix format)
  - LMVM (Limited Memory Variable Metric)
  - Block
  - Nested
  - Dense

- :ref:`Easy, efficient matrix assembly and interface <sec_matcreate>`
- :ref:`Powerful object introspection tools <sec_viewers>`
- Support for a variety of IO options

  .. todo:: find all IO formats
- :ref:`Comprehensive performance testing <ch_performance>`

Solver Features
===============

- :ref:`Parallel Krylov subspace methods <chapter_ksp>`
- :ref:`Parallel nonlinear solvers <chapter_snes>`
- Scalable parallel :ref:`linear <sec_ksppc>` and :ref:`nonlinear <sec_snespc>`
  preconditioners

  .. todo:: port linear solve table
- :ref:`Parallel timestepping (ODE) solvers <chapter_ts>`
- Local and global error estimators
- :ref:`Forward and adjoint sensitivity capabilities <chapter_sa>`

Accelerator/GPU Features
========================

- :ref:`Matrix/Vector CUDA support <doc_config_accel_cuda>`
- :ref:`Matrix/Vector OpenCL/ViennaCL support <doc_config_accel_opencl>`
- Matrix/Vector HIP support

  .. todo:: add HIP documentation
- :ref:`Kokkos support <doc_config_accel_kokkos>`

Support Features
================

- Complete documentation
- :ref:`Comprehensive profiling of floating point and memory usage <ch_profiling>`
- Consistent user interface
- :ref:`Intensive error checking <sec_errors>`
- Over one thousand examples
- PETSc is supported and will be actively enhanced for many years
