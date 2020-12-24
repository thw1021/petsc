.. _doc_gpu_roadmap:

*******************
GPU Support Roadmap
*******************

PETSc algebraic solvers run on GPU systems from NVIDIA using CUDA, and AMD and Intel using
OpenCL/ViennaCL and HIP. Effective GPU implementations of low-level linear algebra
operations provide a highly performant alternative solution strategy for users, and is
therefore a high priority for PETSc developers.

.. note::

   PETSc uses a single source programming model where solver back-ends are selected as
   **runtime** options and configuration options with no changes to the API.

   I.e. users should (ideally) never have to change their source code to take advantage of
     new backend implementations.

PETSc code will include full implementations of vector and matrix operations (as well as
other select operations) using each of:

.. list-table::
   :header-rows: 1

   * - Language/Programming Model
     - Supporting Package
     - Status
   * - CUDA
     - cuBLAS/cuSparse
     - :greenhl:`SUPPORTED`
   * - HIP
     - Rocm
     - :yellowhl:`IN DEVELOPMENT`
   * - SYCL
     - MKL
     - :yellowhl:`IN DEVELOPMENT`
   * - OpenCL
     - ViennaCL
     - :greenhl:`SUPPORTED`
   * - Kokkos
     -
     - :yellowhl:`IN DEVELOPMENT`

---------------------------------

.. admonition:: Important
   :class: yellow

   We could use your help in further developing PETSc for GPUs; see PETSc Developers
   :ref:`documentation <ind_developers>`. The label ``GPU`` is used on our `Gitlab
   <https://gitlab.com/petsc/petsc>`__ repository for all activity involving GPUs.

   **You must use petsc master (git branch) for GPUs, do not install the current release.**
