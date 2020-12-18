.. _doc_faq:

********************************
Frequently Asked Questions (FAQ)
********************************

This page provides help with the most common questions about PETSc, it's design,
execution, and general organization.

.. contents::
   :local:
   :backlinks: top

--------------------------------------------------

=======
General
=======

How can I subscribe to the PETSc mailing lists?
-----------------------------------------------

See mailing list :ref:`documenation <doc_mail>`

Any useful books on numerical computing?
----------------------------------------

`Bueler, PETSc for Partial Differential Equations: Numerical Solutions in C and Python
<https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137>`__

`Writing Scientific Software: A Guide to Good Style
<https://www.mcs.anl.gov/core/books/writing-scientific-software/23206704175AF868E43FE3FB399C2F53>`__

What kind of parallel computers or clusters are needed to use PETSc? Or why do I get little speedup?
----------------------------------------------------------------------------------------------------

.. Important::

   PETSc can be used with any kind of parallel system that supports MPI BUT for any decent
   performance one needs:

   - Fast, **low-latency** interconnect; any ethernet (even 10 GigE) simply cannot provide
     the needed performance.

   - High per-CPU **memory** performance. Each CPU (core in multi-core systems) needs to
     have its **own** memory bandwith of at least 2 or more gigabytes/second. Most modern
     computers are no longer bottlenecked by how fast they can perform a single
     calculation, rather, they are usually restricted by how quickly they can get their
     data.

To obtain good performance it is important that you know your machine! I.e. how many
compute nodes (generally, how many motherboards), how many memory sockets per node and how
many cores per memory socket and how much memory bandwidth for each.

If you do not know this and can run MPI programs with mpiexec (that is, you don't have
batch system), run the following from ``$PETSC_DIR``:

.. code-block:: console

   > make streams [NPMAX=maximum_number_of_mpi_processes_you_plan_to_use]


This will provide a summary of the bandwidth received with different number of MPI
processes and potential speedups. If you have a batch system:

.. code-block:: console

   > cd $PETSC_DIR/src/benchmarks/streams
   > make MPIVersion
   submit MPIVersion to the batch system a number of times with 1, 2, 3, etc MPI processes
   collecting all of the output from the runs into the single file scaling.log. Copy
   scaling.log into the src/benchmarks/streams directory.
   > ./process.py createfile ; process.py

Even if you have enough memory bandwidth if the OS switches processes between cores
performance can degrade. Smart process to core/socket binding (this just means locking a
process to a particular core or memory socket) may help you. For example, consider using
fewer processes than cores and binding processes to separate sockets so that each process
uses a different memory bus:

- `MPICH2 binding with the Hydra process manager
  <https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager#Process-core_Binding>`__

  .. code-block:: console

     > mpiexec.hydra -n 4 --binding cpu:sockets

- `Open MPI binding <http://www.open-mpi.org/doc/v1.5/man1/mpiexec.1.php#sect8>`__

  .. code-block:: console

     > mpiexec -n 4 --bysocket --bind-to-socket --report-bindings

- ``taskset``, part of the `util-linux <https://github.com/karelzak/util-linux>`__ package

  - Check ``man taskset`` for details. Make sure to set affinity for **your** program,
    **not** for the ``mpiexec`` program.

- ``numactl``

  In addition to task affinity, this tool also allows changing the default memory affinity
  policy. On Linux, the default policy is to attempt to find memory on the same memory bus
  that serves the core that a thread is running on at whatever time the memory is faulted
  (not when ``malloc()`` is called). If local memory is not available, it is found
  elsewhere, possibly leading to serious memory imbalances.

  The option ``--localalloc`` allocates memory on the local NUMA node, similar to the
  ``numa_alloc_local()`` function in the ``libnuma`` library. The option
  ``--cpunodebind=nodes`` binds the process to a given NUMA node (note that this can be
  larger or smaller than a CPU (socket); a NUMA node usually has multiple cores).

  The option ``--physcpubind=cpus`` binds the process to a given processor core (numbered
  according to ``/proc/cpuinfo``, therefore including logical cores if Hyper-threading is
  enabled).

  With Open MPI, you can use knowledge of the NUMA hierarchy and core numbering on your
  machine to calculate the correct NUMA node or processor number given the environment
  variable ``OMPI_COMM_WORLD_LOCAL_RANK``. In most cases, it is easier to make mpiexec or
  a resource manager set affinities.

The software `Open-MX <http://open-mx.gforge.inria.fr>`__ provides faster speed for
ethernet systems, we have not tried it but it claims it can dramatically reduce latency
and increase bandwidth on Linux system. You must first install this software and then
install MPICH or Open MPI to use it.

What kind of license is PETSc released under?
---------------------------------------------

See licensing :ref:`documentation <doc_license>`

Why is PETSc written in C, instead of Fortran or C++?
-----------------------------------------------------

C enables us to build data structures for storing sparse matrices, solver information,
etc. in ways that Fortran simply does not allow. ANSI C is a complete standard that all
modern C compilers support. The language is identical on all machines. C++ is still
evolving and compilers on different machines are not identical. Using C function pointers
to provide data encapsulation and polymorphism allows us to get many of the advantages of
C++ without using such a large and more complicated language. It would be natural and
reasonable to have coded PETSc in C++; we opted to use C instead.

Does all the PETSc error checking and logging reduce PETSc's efficiency?
------------------------------------------------------------------------

No

How do such a small group of people manage to write and maintain such a large and marvelous package as PETSc?
-------------------------------------------------------------------------------------------------------------

#. **We work very efficiently**.

   - We use Emacs for all editing (and strongly *discourage* our developers from using
     other editors). Having a uniform editing environment speeds development up immensely.

   - Our manual pages are generated automatically from formatted comments in the code,
     thus alleviating the need for creating and maintaining manual pages.

   - We employ automatic nightly tests of the entire PETSc library on several different
     machine architectures. This process **significantly** protects (no bug-catching
     process is perfect) against inadvertently introducing bugs with new additions. Every
     new feature **must** pass our suite of hundreds of tests as well as formal code
     review before it may be included.

#. **We are very careful in our design (and are constantly revising our design)**

   - PETSc as a package should be easy to use, write, and maintain. Our mantra is to write
     code like everyone is using it.

#. **We are willing to do the grunt work**

   - PETSc is regularly checked to make sure that all code conforms to our interface
     design. We will never keep in a bad design decision simply because changing it will
     require a lot of editing; we do a lot of editing.

#. **We constantly seek out and experiment with new design ideas**

   - We retain the useful ones and discard the rest. All of these decisions are based not
     just on performance, but also on **practicality**.

#. **Function and variable names must adhere to strict guidelines**

   - Even the rules about capitalization are designed to make it easy to figure out the
     name of a particular object or routine. Our memories are terrible, so careful
     consistent naming puts less stress on our limited human RAM.

#. **The PETSc directory tree is carefully designed to make it easy to move throughout the
   entire package**

#. **We have a rich, robust, and fast bug reporting system**,

   - petsc-maint@mcs.anl.gov is always checked, and we pride ourselves on responding
     quickly and accurately. Email is very lightweight, and so bug reports system retains
     an archive of all reported problems and fixes, so it is easy to re-find fixes to
     previously discovered problems.

#. **We contain the complexity of PETSc by using powerful object-oriented programming
   techniques**

   - Data encapsulation serves to abstract complex data formats or movement to
     human-readable format. This is why your program cannot, for example, look directly
     at what is inside the object ``Mat``.

   - Polymorphism makes changing program behavior as easy as possible, and further
     abstracts the *intent* of your program from what is *written* in code. You call
     ``MatMult()`` regardless of whether your matrix is dense, sparse, parallel or
     sequential; you don't call a different routine for each format.

#. **We try to provide the functionality requested by our users**

#. **There is evil here that does not sleep, and the Great Eye is ever watchful**

For complex numbers will I get better performance with C++?
-----------------------------------------------------------

To use PETSc with complex numbers you mayuse the following ``configure`` options;
``--with-scalar-type=complex`` and either ``--with-clanguage=c++`` or (the default)
``--with-clanguage=c``. In our experience they will deliver very similar performance
(speed), but if one is concerned they should just try both and see if one is faster.

How come when I run the same program on the same number of processes I get a "different" answer?
------------------------------------------------------------------------------------------------

Inner products and norms in PETSc are computed using the ``MPI_Allreduce()`` command. For
different runs the order at which values arrive at a given process (via MPI) can be in a
different order, thus the order in which some floating point arithmetic operations are
performed will be different. Since floating point arithmetic arithmetic is not
associative, the computed quantity may be slightly different.

Over a run the many slight differences in the inner products and norms will effect all the
computed results. It is important to realize that none of the computed answers are any
less right or wrong (in fact the sequential computation is no more right then the parallel
ones). All answers are equal, but some are more equal than others.

The discussion above assumes that the exact same algorithm is being used on the different
number of processes. When the algorithm is different for the different number of processes
(almost all preconditioner algorithms except Jacobi are different for different number of
processes) then one expects to see (and does) a greater difference in results for
different numbers of processes. In some cases (for example block Jacobi preconditioner) it
may be that the algorithm works for some number of processes and does not work for others.

How come when I run the same linear solver on a different number of processes it takes a different number of iterations?
------------------------------------------------------------------------------------------------------------------------

The convergence of many of the preconditioners in PETSc including the default parallel
preconditioner block Jacobi depends on the number of processes. The more processes the
(slightly) slower convergence it has. This is the nature of iterative solvers, the more
parallelism means the more "older" information is used in the solution process hence
slower convergence.

Can PETSc use GPUs to speedup computations?
-------------------------------------------

.. note::

   See GPU development :ref:`roadmap <doc_gpu_roadmap>` for the latest information
   regarding the state of PETSc GPU integration.

   See GPU install :ref:`documentation <doc_config_accel>` for up-to-date information
   on installing PETSc to use GPU's.

Recent releases of PETSc have support for running portions of the computation on GPUs. As
GPU support is rapidly evolving target, however, we suggest using the PETSc developer
repository for serious work with GPUs.

PETSc has ``Vec`` classes ``VECCUDA`` and ``VECVIENNACL``, which perform almost all the
vector operations on the GPU. The ``Mat`` classes ``MATAIJCUSPARSE`` and
``MATAIJVIENNACL`` perform matrix-vector products on the GPU but do not support matrix
assembly on the GPU yet.

.. todo:: Stefano? Mark?

Both of these classes run in parallel with MPI. All ``KSP`` methods, except ``KSPIBCGS``,
run all their vector operations on the GPU; thus, for example, Jacobi preconditioned
Krylov methods run completely on the GPU. Preconditioners are a problem; we could do with
some help for these. The example ``src/snes/tutorials/ex47cu.cu`` demonstates how the
nonlinear function evaluation can be done on the GPU.

We plan a significant refactorization of the GPU code in the near future that will make it
easier to use and easier to extend. There is no need to wait for that refactorization
however since the user API and interaction with the GPU will be almost identical. Please
:ref:`contact us <doc_mail>` if you would like examples for a particular programming
model, such as Kokkos, OpenMP, etc. **We will prioritize based on user input**.

.. _doc_faq_extendedprecision:

Can I run PETSc with extended precision?
----------------------------------------

Yes, with gcc 4.6 and later (and gfortran 4.6 and later). ``configure`` PETSc using the
options ``--with-precision=__float128`` and `` --download-f2cblaslapack``.

.. admonition:: Warning
   :class: yellow

   External packages are not guaranteed to work in this mode!

Why doesn't PETSc use QD to implement support for exended precision?
--------------------------------------------------------------------

We tried really hard but could not. The problem is that the QD c++ classes, though they
try to, implement the built-in data types of ``double`` are not native types and cannot
"just be used" in a general piece of numerical source code. Ratherm the code has to
rewritten to live within the limitations of QD classes. However PETSc can be built to use
quad precision, as detailed :ref:`here <doc_faq_extendedprecision>`.

--------------------------------------------------

============
Installation
============
