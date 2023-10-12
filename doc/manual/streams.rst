:html_theme.sidebar_secondary.remove: true

.. _ch_streams:

STREAMS: Example Study
----------------------
Most of the algorithms in PETSc are memory
bandwidth limited. The speed of a simulation depends more on the total achievable [#achievable_footnote]_ memory bandwidth of the computer than the speed
(or number) of floating point units.
The STREAMS benchmark :cite:`streams` is useful to understand parallel performance (scaling) on a shared memory node by measuring achievable memory bandwidth.
PETSc contains
two implementations of the ``triad`` STREAMS benchmark: a `OpenMP version <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/benchmarks/streams/OpenMPVersion.c.html>`__ and a
`MPI version <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/benchmarks/streams/OpenMPVersion.c.html>`__.

.. code-block::

   for (int j = 0; j < n; ++j) a[j] = b[j]+scalar*c[j]

STREAMS measures the total memory bandwidth achieved when running ``n`` independent threads or processes on independent memory regions of an array of total length
``N`` on a shared memory node. The PETSc OpenMP implementation is

.. code-block::

   for (int k = 0; k < NTIMES; K++) {
     time[k] = start clock
     #pragma omp parallel for schedule(static)
     for (int i = 0; i < N; ++i) a[j][i] = b[j][i]+scalar*c[j][i]
     time[k] = end clock - time[k]

The bandwidth is then computed as ``3*N*sizeof(double)/min(time[])``. The timing is done with ``MPI_Wtime()``. A call to this routine takes less than 3e-08 which is significantly
smaller than the operations being timed in this study.

As more threads or processes are added the bandwidth begins to to saturate at some ``n``, generally less than the number of cores on the node. How quickly the bandwidth
saturates and the amount of speed up obtained indicates the likely performance of memory bandwidth limited computations.

Fig. :any:`fig_gcc_streams` plots the total memory bandwidth achieved and the speedup for runs on an Intel system whose details are provided below. The achieved bandwidth
increases rapidly with more cores initially but then less so as more cores are utilized. Also note the improvement may, un-intuitively, be non-monotone when adding
more cores. This is due to the complex inter-connect between the cores and their various levels of caches as well as how the threads or processes are assigned to cores.

.. figure:: /images/manual/gcc_streams.svg
  :alt: STREAMS benchmark gcc
  :name: fig_gcc_streams

  STREAMS benchmark gcc

The STREAMS benchmark is intentionally embarrassingly parallel, that is, each thread or process works on its own data, completely independently of other threads or processes data.
Though real simulations have more complex memory access patterns, most computations for PDEs have large sections of private data and share only data along ghost (halo) regions. Thus the completely
independent STREAMS model still provides useful information.


There are two important concepts needed to understand bandwidth limited computing.

- Thread or process **binding** to hardware subsets of the shared memory node. The Unix operating system allows threads and processes to migrate among the cores of a node
  during a computation. This migration is managed by the operating system (OS). [#memorymigration_footnote]_
  A thread or process that is "near" some data may suddenly be far from the data when the thread or process gets migrated.
  Binding the thread or process to a hardware unit prevents this.

- Thread or process **mapping** (assignment) to hardware subsets when more threads or processes are used. Physical memory is divided into multiple distinct units, each of which can
  independently provide a certain memory bandwidth. When increasing from one thread or process to two one obviously would like the second thread
  or process to use a different memory unit
  and hence not share the same unit with the first thread or process.

  Different cores may be more closely connected to different memory units. This results in
  non-uniform memory access (**NUMA**) which means the memory latency or bandwidth for any particular core depends on the physical address of the requested memory.
  Mapping each new thread or process to cores that do not share the previously assigned core's memory unit ensures a higher total achievable bandwidth.

  In addition to mapping, one must ensure that each thread or process **actually uses the closest memory unit**. The OS selects the memory unit based on **first touch**,
  the core of the first thread or process to touch (read or write) a memory address determines to which memory unit the data is assigned. For multiple processes this is automatic
  since only that one process (on a particular core) will ever touch its data. For threads care must be taken that the data a thread is to compute on is first touched by that thread.
  However, for small data arrays that remain in cache first touch may produce no difference in performance.

MPI and OpenMP provide ways to bind and map processes and cores. They also provide ways to display the current mapping.

- MPI, options to ``mpiexec``

  - --bind-to hwthread | core | l1cache | l2cache | l3cache | socket | numa | board

  - --map-by hwthread | core | socket | numa | board | node

  - --report-bindings

  - --cpu-list list of cores

  - --cpu-set list of sets of cores

- OpenMP

  - OMP_NUM_THREADS=close | spread

  - OMP_PROC_BIND=close | spread

  - OMP_PLACES="list of sets of cores" for example {0:2},{2:2},{32:2},{34:2}

  - OMP_DISPLAY_ENV=false | true

  - OMP_DISPLAY_AFFINITY=false | true

Providing appropriate values may be crucial to high performance; the defaults may produce poor results. The best options for the STREAMS benchmark may be the best options for large PETSc applications.

Detailed STREAMS study for large arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now present a detailed study of a particular Intel Icelake system, the Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GH. It has 32 cores on each of two sockets
(each with a single NUMA region, so a total of two NUMA regions), a
48 Megabyte L3 cache and 32 1.25 Megabyte L2 caches, each shared by 2 cores.
It is running the Rocky Linux 8.8 (Green Obsidian) distribution. The compilers
used are GNU 12.2, Intel(R) oneAPI Compiler 2023.0.0 with both icc and icx, and NVIDIA nvhpc/23.1. The MPI implementation is OpenMPI 4.0.7, except for nvhpc which uses 3.15.. The compiler options were

- gcc -O3 -march=native

- icc -O3 -march=native

- icx -O3 -ffinite-math-only (the -xHost option, that replaces -march=native, crashed the compiler so was not used)

- nvc -O3 -march=native

We first run the STREAMS benchmark with large double precision arrays of length :math:`1.6\times10^8`; the size was selected to be large enough to eliminate cache effects.
Fig. :any:`fig_streams` shows the achieved bandwidth for gcc, icc, icx, and nvc using MPI and OpenMP with their default bindings and with the MPI binding of ``--bind-to core --map-by numa``
and the OpenMP binding of ``OMP_PROC_BIND=spread``.

.. figure:: /images/manual/streams.svg
  :alt: STREAMS benchmark
  :name: fig_streams

  Comprehensive STREAMS performance on Intel system

Note the two dips in the performance with OpenMP and gcc using binding in Fig. :any:`fig_gcc_streams`.
Requesting the ``spread`` binding produces better results for small core counts, but poorer results for larger core counts.
These are a result of a bug in the gcc ``spread`` option placing more threads in one NUMA domain than the other.
For example, the ``OMP_DISPLAY_AFFINITY`` shows that for 28 threads, 12 are placed on NUMA region 1 and 16 are placed on the other NUMA region.

Fig. :any:`fig_icc_streams` shows the performance with the icc compiler. Note that the icc compiler produces significantly faster code than the other compilers so its STREAMS speedups are smaller,
though it
provides better performance. No significant dips occur with the OpenMP binding using icc, icx, and nvc;
using ``OMP_DISPLAY_AFFINITY`` confirms, for example, that 14 threads (out of 28) are assigned to each NUMA domain unlike with gcc.
Using the exact thread placement that icc uses with gcc using the OpenMP ``OMP_PLACES`` option removes most of the dip in the gcc OpenMP binding result.
Thus we conclude that on this system the ``spread`` option does not always give the best thread placement with gcc.

.. figure:: /images/manual/icc_streams.svg
  :alt: STREAMS benchmark icc
  :name: fig_icc_streams

  STREAMS benchmark icc

Fig. :any:`fig_icx_streams` shows the performance with the icx compiler.

.. figure:: /images/manual/icx_streams.svg
  :alt: STREAMS benchmark icx
  :name: fig_icx_streams

  STREAMS benchmark icx

.. figure:: /images/manual/nvc_streams.svg
  :alt: STREAMS benchmark nvc
  :name: fig_nvc_streams

  STREAMS benchmark nvc

Observations:

- For MPI the default binding and mapping on this system produces results as good as providing a specific binding and mapping.

- For OpenMP gcc, the default binding is better than using ``spread``, because ``spread`` has a bug. For the other compilers using ``spread`` is crucial for good performance on more than 32 cores.

- We do not have any explanation why the improvement in speedup for gcc, icx, and nvc slows down between 32 and 48 cores and then improves rapidly since we believe appropriate bindings are being used.

Detailed STREAMS study for small arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall the global array size above was selected to be large enough to minimize cache effects. We now consider runs with very small arrays of size :math:`120,000`. We provide similar
plots to those given above.

.. figure:: /images/manual/smallstreams.svg
  :alt: Small STREAMS benchmark
  :name: fig_small_streams

  Comprehensive small STREAMS performance on Intel system

.. figure:: /images/manual/gcc_smallstreams.svg
  :alt: SmallSTREAMS benchmark gcc
  :name: fig_gcc_smallstreams

  Small STREAMS benchmark gcc

.. figure:: /images/manual/icc_smallstreams.svg
  :alt: SmallSTREAMS benchmark icc
  :name: fig_icc_smallstreams

  Small STREAMS benchmark icc

.. figure:: /images/manual/icx_smallstreams.svg
  :alt: SmallSTREAMS benchmark icx
  :name: fig_icx_smallstreams

  Small STREAMS benchmark icx

In Fig. :any:`fig_gcc_openmp_smallstreams` we plot only the results for OpenMP with gcc. Note that the explicit binding makes no difference and the results are much poorer than icc
(Fig. :any:`fig_icc_smallstreams` ) and icx (Fig. :any:`fig_icx_smallstreams` ), but both explicit binding and no binding results decrease sharply after 32 cores; much like the icc and
icx results **without** binding. We have no explanation for the terrible behavior of gcc with OpenMP with small array sizes.

While running the benchmark for MPI the results became inconsistent, for a particular number of MPI processes the performance drops by a factor of two, but then returns to the higher
values at a later number of processes. In another run a different number of MPI processes may produce the lower values. We do not have an explanation for this behavior. It is possibly
related to clock speed throttling but we have no particular reason to believe this.

We have no explanation for why the **OpenMP version is consistently
much worse than the MPI version for such small arrays**.


.. figure:: /images/manual/gcc_openmp_smallstreams.svg
  :alt: SmallSTREAMS benchmark gcc, only OpenMP
  :name: fig_gcc_openmp_smallstreams

  Small STREAMS benchmark gcc, only OpenMP


To understand better the poor scaling of the OpenMP STREAMS for small arrays we have created `a new benchmark code <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/benchmarks/streams/OpenMPVersionLikeMPI.c.html>`__.
This code uses different arrays for each thread instead of having the threads share non-overlapping regions of the same array. The performance of this code is the same as the traditional OpenMP code.
However it offers the possibility of adding another loop immediately around the computational loop

.. code-block::

   for (int k = 0; k < NTIMES; K++) {
     start clock
     #pragma omp parallel for schedule(static)
     for (int j = 0; j < size; j++)
       n = (N / size + ((N % size) > omp_get_thread_num()))
       for (int l=0; l<NTimesInner; l++)
         for (int i = 0; i < n; ++i) a[j][i] = b[j][i]+scalar*c[j][i]
     end clock

where ``size`` is the number of threads and the sum of the ``n`` over all threads is ``N``. This same inner loop was be added to the MPI version.

The results for the icx compiler are displayed in Fig.
:any:`fig_icx_inner_smallstreams`. For smaller number of threads the OpenMP version now matches the MPI STREAM version (the MPI version at this scale has the same results with and without the inner loop).
For larger thread counts both the OpenMP and MPI versions with the inner loop continue to behave similarly and the performance is better than plain MPI version. For gcc and icc the behavior is
dramatically different, we have no explanation for the difference.

.. figure:: /images/manual/gcc_inner_smallstreams.svg
  :alt: Small STREAMS benchmark gcc and inner loop
  :name: fig_gcc_inner_smallstreams

  Small STREAMS benchmark gcc and inner loop

.. figure:: /images/manual/icc_inner_smallstreams.svg
  :alt: Small STREAMS benchmark icc and inner loop
  :name: fig_icc_inner_smallstreams

  Small STREAMS benchmark icc and inner loop

.. figure:: /images/manual/icx_inner_smallstreams.svg
  :alt: Small STREAMS benchmark icx and inner loop
  :name: fig_icx_inner_smallstreams

  Small STREAMS benchmark icx and inner loop

.. figure:: /images/manual/nvc_inner_smallstreams.svg
  :alt: Small STREAMS benchmark nvc and inner loop
  :name: fig_nvc_inner_smallstreams

  Small STREAMS benchmark nvc and inner loop

Detailed STREAMS study for medium size arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We next examine the performance for arrays of size 1,200,000 and produce similar figures. Again there is dramatically different performance for different compilers.

.. figure:: /images/manual/mediumstreams.svg
  :alt: medium STREAMS benchmark
  :name: fig_medium_streams

  Comprehensive medium STREAMS performance on Intel system

.. figure:: /images/manual/gcc_inner_mediumstreams.svg
  :alt: Medium STREAMS benchmark gcc and inner loop
  :name: fig_gcc_inner_mediumstreams

  Medium STREAMS benchmark gcc and inner loop

.. figure:: /images/manual/icc_inner_mediumstreams.svg
  :alt: Medium STREAMS benchmark icc and inner loop
  :name: fig_icc_inner_mediumstreams

  Medium STREAMS benchmark icc and inner loop

.. figure:: /images/manual/icx_inner_mediumstreams.svg
  :alt: Medium STREAMS benchmark icx and inner loop
  :name: fig_icx_inner_mediumstreams

  Medium STREAMS benchmark icx and inner loop

.. figure:: /images/manual/nvc_inner_mediumstreams.svg
  :alt: Medium STREAMS benchmark nvc and inner loop
  :name: fig_nvc_inner_mediumstreams

  Medium STREAMS benchmark nvc and inner loop

.. table:: Maximum bandwidths; small, medium, and large
   :name: tab-1

   ==================== =========== =========== ======= ==========
   Compiler             OpenMP       with inner MPI     with inner
   ==================== =========== =========== ======= ==========
   .                    .           Small       .       .
   gcc                  .5*         7           7       11
   icc                  1.5*        7.5         5.5     8.5
   icx                  1.5*        1*          6       10.2
   nvc                  1*          2*          6       10
   .                    .           Medium      .       .
   gcc                  2           7.5         7.5     7.5
   icc                  3.5         5           5       5
   icx                  .5*         7.5         1*      1*
   nvc                  3.5         2*          6.5     6.5
   .                    .           Large       .       .
   gcc                  .3           .          .3      .
   icc                  .3           .          .35     .
   icx                  .3           .          .3      .
   nvc                  .3           .          .3      .
   ==================== =========== =========== ======= ==========

In Table :any:`tab-1` we present the bandwidths achieved at 64 cores for all the compilers with binding on the small, medium, and large cases. We have mark the surprisingly small values with an \*.

Observations:

- Different compilers often resulted in very different performance when working with small and moderate sized arrays. While with large arrays (except with icc using full optimization) the results are
  very similar for all compilers.


Detailed study with simple application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now move on to a `PETSc application <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex45.c.html>`__ which solves a three dimensional Poisson problem on a unit squire discretized with
finite differences whose linear system is solved with the PETSc algebraic multigrid, `PCGAMG`. To match the STREAMS benchmark, strong scaling is used measuring the time to construct the preconditioner,
the time to solve the linear system with the preconditioner, and the time for all the matrix-vector products. These are displayed in Fig. :any:`fig_gamg`. The runtime options were
``-da_refine 6 -pc_type gamg -log_view``. In this study there was no attempt to tune the default ``PCGAMG`` parameters.
Note the similar speedups for all the
different compilers. Also note the solver and matrix-vector multiplication (which is part of the solver) have **speedup results remarkably close to the large STREAMS benchmark**, but without the slower
growth in speedup between 32 and 48 cores exhibited by STREAMS. Even the icc compiler version produces speedups in the 25 range while for the STREAMS benchmark it is around 17;
we conclude that this is because none of the PETSc code has the same "extra" performance from icc that the benchmark has.

.. figure:: /images/manual/gamg.svg
  :alt: GAMG solver speedup
  :name: fig_gamg

  GAMG Solver Speedup

To understand the disparity in the STREAMS performance with icc we reran it with the highest optimization level that produced the same results as gcc and icx: ``-O1`` without ``-march=native``.
The results are displayed in Fig. :any:`fig_icc_O1_streams`, sure enough the results now match that of gcc and icx.

.. figure:: /images/manual/icc_O1_streams.svg
  :alt: STREAMS benchmark icc -O1
  :name: fig_icc_O1_streams

  STREAMS benchmark icc -O1

Next we reran the simple application with the lower icc optimization. The results are in Fig. :any:`fig_gamg_O1` and exhibit the same behavior as the code with full optimization indicating the extra
optimization effects the STREAMS benchmark but not the application code.

.. figure:: /images/manual/gamg_O1.svg
  :alt: GAMG solver speedup icc -O1
  :name: fig_gamg_O1

  GAMG Solver Speedup icc -O1

The speed ups for the ``PC`` setup times are higher than for the solver and the benchmark, this is not surprising since the ``PCGAMG`` setup is not as bandwidth limited as the solve. Also the
setup time continues to have improved speedup well passed 32 cores while the solver time has nearly saturated at 32 cores. Since the setup time dominates this application, using as many cores
as possible will result in a measurably smaller compute time. We have no particular explanation for the dips in the performance at certain core counts but note it is consistent between compilers
and likely results from the amount of MPI communication required, the communication pattern, and also differences in the performance of ``PCGAMG`` which does vary for different core counts
since the algorithms used produce "slightly" different preconditioners.

This example demonstrates the utility of the STREAMS benchmark to predict the speedup of a memory bandwidth limited application on a shared memory system.

.. _sec_pcmpi_study:

Simple application with the MPI linear solver server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now run the same PETSc application but with MPI linear solver server mode; set using the ``-mpi_linear_solver_server`` option.  In Fig. :any:`fig_gamg_server` we
plot the time to solution (both time to distribute the system, the setup and the solve time) was well as, separately the time needed to distribute the system.
All compilers deliver largely the same performance. Note that performance improvement
begins to tail off at around 20 MPI processes. In  Fig. :any:`fig_gamg_server_percent` we plot the percentage of the solver time needed for the matrix (and vector) distributions. It begins at
around 13 percent and grows rapidly at around 20 MPI processes to 30 percent. At 64 MPI processes it is one-half the solution time. Finally we plot the speedup in Fig. :any:`fig_gamg_server_speedup`
and not that it is disappointingly fair below the parallel solve without the server.


.. figure:: /images/manual/gamg_server.svg
  :alt: GAMG solver time  with linear solver server
  :name: fig_gamg_server

  GAMG Solver time with linear solver server

.. figure:: /images/manual/gamg_server_percent.svg
  :alt: GAMG solver percent time in distribution  with linear solver server
  :name: fig_gamg_server_percent

  GAMG Solver percent time in distribution  with  linear solver server

.. figure:: /images/manual/gamg_server_speedup.svg
  :alt: GAMG solver speedup  with linear solver server
  :name: fig_gamg_server_speedup

  GAMG Solver speedup  with  linear solver server 

The initial implementation of ``PCMPI``, benchmarked above, used ``MPI_Scatterv()`` to communicate the matrix and vector entries from the initial compute process to all of the
server processes. Unfortunately ``MPI_Scatterv()`` does not scale with more MPI processes hence the solution time is limited by the ``MPI_Scatterv()``. To remove this limitation
we implemented an alternative communication mechanism where the MPI processes used ``MPI_Win_allocate_shared()`` to allocate a common pool of memory from which all the MPI processes in the server
can access their portion of the matrices and vectors. There is still an extra server processing overhead since the initial data storage of the sequential matrix (in ``MATSEQAIJ`` storage)
still needs to be converted to ``MATMPIAIJ`` storage. Fortunately this conversion scales well as seen in Fig. :any:`fig_gamg_server_shared_percent` remaining at around ten percent
of the solution time. ``VecPlaceArray()`` is used to convert the sequential vector to an MPI vector so there is
no overhead, not even a copy, for this operation. Again we plot the speedup in Fig. :any:`fig_gamg_server_shared_speedup`
and note that it is significantly higher than with the ``MPI_Scatterv()`` based server though still below the parallel solver without the server.

.. figure:: /images/manual/gamg_server_shared.svg
  :alt: GAMG solver time  with linear solver server and shared memory distribution
  :name: fig_gamg_server_shared

  GAMG Solver time with linear solver server and shared memory distribution

.. figure:: /images/manual/gamg_server_shared_percent.svg
  :alt: GAMG solver percent time in distribution  with linear solver server
  :name: fig_gamg_server_shared_percent

  GAMG Solver percent time in distribution  with  linear solver server and shared memory distribution

.. figure:: /images/manual/gamg_server_shared_speedup.svg
  :alt: GAMG solver speedup  with linear solver server and shared memory distribution
  :name: fig_gamg_server_shared_speedup

  GAMG Solver speedup  with  linear solver server and shared memory distribution

.. rubric:: Footnotes

.. [#achievable_footnote] Achievable memory bandwidth is the actual bandwidth one can obtain
   as opposed to the theoretical peak that is calculated using the hardware specification.

.. [#memorymigration_footnote] Data can also be migrated among different memory sockets during a computation by the OS, but we ignore this possibility in the discussion.

.. bibliography:: /petsc.bib
   :filter: docname in docnames
