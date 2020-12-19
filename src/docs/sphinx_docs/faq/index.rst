.. _doc_faq:

==================================
 Frequently Asked Questions (FAQ)
==================================

This page provides help with the most common questions about PETSc, it's design,
execution, and general organization.

.. contents::
   :local:
   :backlinks: top

--------------------------------------------------

General
=======

How Can I Subscribe To The PETSc Mailing Lists?
-----------------------------------------------

See mailing list :ref:`documenation <doc_mail>`

Any Useful Books On Numerical Computing?
----------------------------------------

`Bueler, PETSc for Partial Differential Equations: Numerical Solutions in C and Python
<https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137>`__

`Writing Scientific Software: A Guide to Good Style
<https://www.mcs.anl.gov/core/books/writing-scientific-software/23206704175AF868E43FE3FB399C2F53>`__

What Kind Of Parallel Computers Or Clusters Are Needed To Use PETSc? Or Why Do I Get Little Speedup?
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

What Kind Of License Is PETSc Released Under?
---------------------------------------------

See licensing :ref:`documentation <doc_license>`

Why Is PETSc Written In C, Instead Of Fortran Or C++?
-----------------------------------------------------

C enables us to build data structures for storing sparse matrices, solver information,
etc. in ways that Fortran simply does not allow. ANSI C is a complete standard that all
modern C compilers support. The language is identical on all machines. C++ is still
evolving and compilers on different machines are not identical. Using C function pointers
to provide data encapsulation and polymorphism allows us to get many of the advantages of
C++ without using such a large and more complicated language. It would be natural and
reasonable to have coded PETSc in C++; we opted to use C instead.

Does All The PETSc Error Checking And Logging Reduce PETSc's Efficiency?
------------------------------------------------------------------------

No

How Do Such A Small Group Of People Manage To Write And Maintain Such A Large And Marvelous Package As PETSc?
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

For Complex Numbers Will I Get Better Performance With C++?
-----------------------------------------------------------

To use PETSc with complex numbers you mayuse the following ``configure`` options;
``--with-scalar-type=complex`` and either ``--with-clanguage=c++`` or (the default)
``--with-clanguage=c``. In our experience they will deliver very similar performance
(speed), but if one is concerned they should just try both and see if one is faster.

How Come When I Run The Same Program On The Same Number Of Processes I Get A "Different" Answer?
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

How Come When I Run The Same Linear Solver On A Different Number Of Processes It Takes A Different Number Of Iterations?
------------------------------------------------------------------------------------------------------------------------

The convergence of many of the preconditioners in PETSc including the default parallel
preconditioner block Jacobi depends on the number of processes. The more processes the
(slightly) slower convergence it has. This is the nature of iterative solvers, the more
parallelism means the more "older" information is used in the solution process hence
slower convergence.

Can PETSc Use GPU's To Speedup Computations?
--------------------------------------------

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

Can I Run PETSc With Extended Precision?
----------------------------------------

Yes, with gcc 4.6 and later (and gfortran 4.6 and later). ``configure`` PETSc using the
options ``--with-precision=__float128`` and `` --download-f2cblaslapack``.

.. admonition:: Warning
   :class: yellow

   External packages are not guaranteed to work in this mode!

Why Doesn't PETSc Use Qd To Implement Support For Exended Precision?
--------------------------------------------------------------------

We tried really hard but could not. The problem is that the QD c++ classes, though they
try to, implement the built-in data types of ``double`` are not native types and cannot
"just be used" in a general piece of numerical source code. Ratherm the code has to
rewritten to live within the limitations of QD classes. However PETSc can be built to use
quad precision, as detailed :ref:`here <doc_faq_extendedprecision>`.

--------------------------------------------------

Installation
============

How Do I Begin Using PETSc If The Software Has Already Been Completely Built And Installed By Someone Else?
-----------------------------------------------------------------------------------------------------------

Assuming that the PETSc libraries have been successfully built for a particular
architecture and level of optimization, a new user must merely:

#. Set ``$PETSC_DIR`` to the full path of the PETSc home
   directory. This will be the location of the ``configure`` script, and usually called
   "petsc" or some vairation of that (for example, /home/username/petsc).

#. Set ``$PETSC_ARCH``, which indicates the configuration on which PETSc will be
   used. Note that the ``$PETSC_ARCH`` is simply a name the installer used when installing
   the libraries. There will exist a directory within ``$PETSC_DIR`` that is named after
   its corresponding ``$PETSC_ARCH``. There many be several on a single system, for
   example "linux-c-debug" for the debug versions compiled by a c compiler or
   "linux-c-opt" for the optimized version.

.. admonition:: Still Stuck?

   See the :ref:`quick-start tutorial <tut_install>` for a step-by-step guide on
   installing PETSc, in case you have missed a step.

   See the users manual section on :ref:`getting started <sec-getting-started>`.

The PETSc Distribution Is SO Large. How Can I Reduce My Disk Space Usage?
-------------------------------------------------------------------------

#. The directory ``$PETSC_DIR/docs`` contains a set of HTML manual pages in for use with a
   browser. You can delete these pages to save some disk space.

#. The PETSc users manual is provided in PDF format at ``$PETSC_DIR/docs/manual.pdf``. You
   can delete this.

#. The PETSc test suite contains sample output for many of the examples. These are
   contained in the PETSc directories ``$PETSC_DIR/src/*/tutorials/output`` and
   ``$PETSC_DIR/src/*/tests/output``. Once you have run the test examples, you may remove
   all of these directories to save some disk space.

#. The debugging versions of the libraries are larger than the optimized versions. In a
   pinch you can work with the optimized version, although we bid you good luck in
   finnding bugs as it is much easier with the debug version.

I Want To Use PETSc Only For Uniprocessor Programs. Must I Still Install And Use A Version Of MPI?
--------------------------------------------------------------------------------------------------

No, run ``configure`` with the option ``--with-mpi=0``

Can I Install PETSc To Not Use X Windows (Either Under Unix Or Windows With GCC)?
---------------------------------------------------------------------------------

Yes. Run ``configure`` with the additional flag ``--with-x=0``

Why Do You Use MPI?
-------------------

MPI is the message-passing standard. Because it is a standard, it will not change over
time; thus, we do not have to change PETSc every time the provider of the message-passing
system decides to make an interface change. MPI was carefully designed by experts from
industry, academia, and government labs to provide the highest quality performance and
capability.

For example, the careful design of communicators in MPI allows the easy nesting of
different libraries; no other message-passing system provides this support. All of the
major parallel computer vendors were involved in the design of MPI and have committed to
providing quality implementations.

In addition, since MPI is a standard, several different groups have already provided
complete free implementations. Thus, one does not have to rely on the technical skills of
one particular group to provide the message-passing libraries. Today, MPI is the only
practical, portable approach to writing efficient parallel numerical software.

What Do I Do If My MPI Compiler Wrappers Are Invalid?
-----------------------------------------------------

Most MPI implementations provide compiler wrappers (such as ``mpicc``) which give the
include and link options necessary to use that verson of MPI to the underlying compilers
. These wrappers are either absent or broken in the MPI pointed to by
``--with-mpi-dir``. You can rerun the configure with the additional option
``--with-mpi-compilers=0``, which will try to auto-detect working compilers; however,
these compilers may be incompatible with the particular MPI build. If this fix does not
work, run with ``--with-cc=[your_c_compiller]`` where you know ``your_c_compiler`` works
with this particular MPI, and likewise for C++ and Fortran.

When Should/Can I Use The ``configure`` Option ``--with-64-bit-indices``?
-------------------------------------------------------------------------

By default the type that PETSc uses to index into arrays and keep sizes of arrays is a
``PetscInt`` defined to be a 32 bit ``int``. If your problem:

- Involves more than 2^31 - 1 unknowns (around 2 billion).

- Your matrix might contain more than 2^31 - 1 nonzeros on a single process.

Then you need to use this option. Otherwise you will get strange crashes.

This option can be used when you are using either 32 bit or 64 bit pointers. You do not
need to use this option if you are using 64 bit pointers unless the two conditions above
hold.

What If I Get An Internal Compiler Error?
-----------------------------------------

You can rebuild the offending file individually with a lower optimization level. **Then
make sure to complain to the compiler vendor and file a bug report**. For example, if the
compiler chokes on ``src/mat/impls/baij/seq/baijsolvtrannat.c`` you can run the following
from ``$PETSC_DIR``:

.. code-block:: console

   > make -f gmakefile PCC_FLAGS="-O1" $PETSC_ARCH/obj/src/mat/impls/baij/seq/baijsolvtrannat.o
   > make all

How Do I Install petsc4py With The Development PETSc?
-----------------------------------------------------

#. Install `Cython <https://cython.org/>`__.

#. ``configure`` PETSc with the ``--download-petsc4py`` option.

What Fortran Compiler Do You Recommend On macOS?
------------------------------------------------

We recommend using `homebrew <https://brew.sh/>`__ to install `gfortran
<https://gcc.gnu.org/wiki/GFortran>`__

Please contact Apple at https://www.apple.com/feedback/ and urge them to bundle gfortran
with future versions of Xcode.

How Can I Find The URL Locations Of The Packages You Install Using ``--download-PACKAGE``?
------------------------------------------------------------------------------------------

.. code-block:: console

   > grep "self.download " $PETSC_DIR/config/BuildSystem/config/packages/*.py

How To Fix The Problem: PETSc Was Configured With One MPICH (Or Open MPI) ``mpi.h`` Version But Now Appears To Be Compiling Using A Different MPICH (Or Open MPI) ``mpi.h`` Version
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This happens for generally one of two reasons:

- You previously ran ``configure`` with the option ``--download-mpich`` (or ``--download-openmpi``)
  but later ran ``configure`` to use a version of MPI already installed on the
  machine. Solution:

  .. code-block:: console

     > rm -rf $PETSC_DIR/$PETSC_ARCH
     > ./configure --your-args

What Does It Mean When ``make check`` Errors On ``PetscOptionsInsertFile()``?
-----------------------------------------------------------------------------

For example:

::

   Possible error running C/C++ src/snes/tutorials/ex19 with 2 MPI processes
   See https://www.mcs.anl.gov/petsc/documentation/faq.html
   [0]PETSC ERROR: #1 PetscOptionsInsertFile() line 563 in /Users/barrysmith/Src/PETSc/src/sys/objects/options.c
   [0]PETSC ERROR: #2 PetscOptionsInsert() line 720 in /Users/barrysmith/Src/PETSc/src/sys/objects/options.c
   [0]PETSC ERROR: #3 PetscInitialize() line 828 in /Users/barrysmith/Src/PETSc/src/sys/objects/pinit.c

- You may be using the wrong ``mpiexec`` for the MPI you have linked PETSc with.

- You have VPN enabled on your machine whose network settings may not play well with MPI.

The machine has a funky network configuration and for some reason MPICH is unable to
communicate between processes with the socket connections it has established. This can
happen even if you are running MPICH on just one machine. Often you will find that ``ping
hostname`` fails with this network configuration; that is, processes on the machine cannot
even connect to the same machine. You can try completely disconnecting your machine from
the network and see if ``make check`` then works or speaking with your system
administrator. You can also try the ``configure`` options ``--download-mpich`` or
``--download-mpich-device=ch3:nemesis``.

--------------------------------------------------

Usage
=====

How Can I Redirect PETSc's ``stdout`` And ``stderr`` When Programming With A Gui Interface In Windows Developer Studio Or To C++ Streams?
-----------------------------------------------------------------------------------------------------------------------------------------

To overload just the error messages write your own ``MyPrintError()`` function that does
whatever you want (including pop up windows etc) and use it like below.

.. code-block:: c

   extern "C" {
     int PASCAL WinMain(HINSTANCE,HINSTANCE,LPSTR,int);
   };

   #include <petscsys.h>
   #include <mpi.h>

   const char help[] = "Set up from main";

   int MyPrintError(const char error[], ...)
   {
     printf("%s", error);
     return 0;
   }

   int main(int ac, char *av[])
   {
     char           buf[256];
     HINSTANCE      inst;
     PetscErrorCode ierr;

     inst = (HINSTANCE)GetModuleHandle(NULL);
     PetscErrorPrintf = MyPrintError;

     buf[0] = 0;
     for (int i = 1; i < ac; ++i) {
       strcat(buf, av[i]);
       strcat(buf, " ");
     }

     ierr = PetscInitialize(&ac, &av, NULL, help);if (ierr) return ierr;

     return WinMain(inst, NULL, buf, SW_SHOWNORMAL);
   }

Place this file in the project and compile with this preprocessor definitions:

::

   WIN32
   _DEBUG
   _CONSOLE
   _MBCS
   USE_PETSC_LOG
   USE_PETSC_BOPT_g
   USE_PETSC_STACK
   _AFXDLL

And these link options:

::

   /nologo
   /subsystem:console
   /incremental:yes
   /debug
   /machine:I386
   /nodefaultlib:"libcmtd.lib"
   /nodefaultlib:"libcd.lib"
   /nodefaultlib:"mvcrt.lib"
   /pdbtype:sept

.. note::

   The above is compiled and linked as if it was a console program. The linker will search
   for a main, and then from it the ``WinMain`` will start. This works with MFC templates and
   derived classes too.

   When writing a Window's console application you do not need to do anything, the ``stdout``
   and ``stderr`` is automatically output to the console window.

To change where all PETSc ``stdout`` and ``stderr`` go, (you can also reassign
``PetscVFPrintf()`` to handle ``stdout`` and ``stderr`` any way you like) write the
following function:

::

   PetscErrorCode mypetscvfprintf(FILE *fd, const char format[], va_list Argp)
   {
     PetscErrorCode ierr;

     PetscFunctionBegin;
     if (fd != stdout && fd != stderr) { /* handle regular files */
       ierr = PetscVFPrintfDefault(fd, format, Argp);CHKERRQ(ierr);
     } else {
       char buff[1024]; /* Make sure to assign a large enough buffer */
       int  length;

       ierr = PetscVSNPrintf(buff, 1024, format, &length, Argp);CHKERRQ(ierr);

       /* now send buff to whatever stream or whatever you want */
     }
     PetscFunctionReturn(0);
   }

Then assign ``PetscVFPrintf = mypetscprintf`` before ``PetscInitialize()`` in your main
program.

I Want To Use Hypre boomerAMG Without GMRES But When I Run ``-pc_type hypre -pc_hypre_type boomeramg -ksp_type preonly`` I Don't Get A Very Accurate Answer!
------------------------------------------------------------------------------------------------------------------------------------------------------------

You should run with ``-ksp_type richardson`` to have PETSc run several V or W
cycles. ``-ksp_type preonly`` causes boomerAMG to use only one V/W cycle. You can control
how many cycles are used in a single application of the boomerAMG preconditioner with
``-pc_hypre_boomeramg_max_iter <it>`` (the default is 1). You can also control the
tolerance boomerAMG uses to decide if to stop before ``max_iter`` with
``-pc_hypre_boomeramg_tol <tol>`` (the default is 1.e-7). Run with ``-ksp_view`` to see
all the hypre options used and ``-help | grep boomeramg`` to see all the command line
options.

How Do I Use PETSc For Domain Decomposition?
--------------------------------------------

PETSc includes Additive Schwarz methods in the suite of preconditioners under the umbrella
of ``PCASM``. These may be activated with the runtime option ``-pc_type asm``. Various
other options may be set, including the degree of overlap ``-pc_asm_overlap <number>`` the
type of restriction/extension ``-pc_asm_type [basic,restrict,interpolate,none]`` sets ASM
type and several others. You may see the available ASM options by using ``-pc_type asm
-help``. See the procedural interfaces in the manual pages, for example ``PCASMType()``
and check the index of the users manual for ``PCASMCreateSubDomains()``.

PETSc also contains a domain decomposition inspired wirebasket or face based two level
method where the coarse mesh to fine mesh interpolation is defined by solving specific
local subdomain problems. It currently only works for 3D scalar problems on structured
grids created with PETSc ``DMDA``. See the manual page for ``PCEXOTIC`` and
``src/ksp/ksp/tutorials/ex45.c`` for an example.

PETSc also contains a balancing Neumann-Neumann type preconditioner, see the manual page
for ``PCBDDC``. This requires matrices be constructed with ``MatCreateIS()`` via the finite
element method. See ``src/ksp/ksp/tests/ex59.c`` for an example.

You Have AIJ And BAIJ Matrix Formats, And SBAIJ For Symmetric Storage, How Come No SAIJ?
----------------------------------------------------------------------------------------

Just for historical reasons; the SBAIJ format with blocksize one is just as efficient as
an SAIJ would be.

Can I Create BAIJ Matrices With Different Size Blocks For Different Block Rows?
-------------------------------------------------------------------------------

No. The ``MATBAIJ`` format only supports a single fixed block size on the entire
matrix. But the ``MATAIJ`` format automatically searches for matching rows and thus still
takes advantage of the natural blocks in your matrix to obtain good performance.

.. note::

   If you use ``MATIJ`` you cannot use the ``MatSetValuesBlocked()``.

How Do I Access The Values Of A Remote Parallel PETSc ``Vec``?
--------------------------------------------------------------

#. On each process create a local ``Vec`` large enough to hold all the values it wishes to
   access.

#. Create a ``VecScatter`` that scatters from the parallel ``Vec`` into the local ``Vec``.

#. Use ``VecGetArray()`` to access the values in the local ``Vec``.

.. _doc_faq_usage_alltoone:

How Do I Collect To A Single Processor All The Values From A Parallel PETSc ``Vec``?
------------------------------------------------------------------------------------

#. Create the ``VecScatter`` context that will do the communication:

   ::

      Vec            in_par, out_seq;
      VecScatter     ctx;
      PetscErrorCode ierr;

      ierr = VecScatterCreateToAll(in_par, &ctx, &out_seq);CHKERRQ(ierr);

#. Initiate the communication (this may be repeated if you wish):

   ::

      ierr = VecScatterBegin(ctx, in_par, out_seq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ctx, in_par, out_seq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
      /* May destroy context now if no additional scatters are needed, otherwise reuse it */
      ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

Note that this simply concatenates in the parallel ordering of the vector (computed by the
``MPI_Comm_rank`` of the owning process). If you are using a ``Vec`` from
``DMCreateGlobalVector()`` you likely want to first call ``DMDAGlobalToNaturalBegin()``
followed by ``DMDAGlobalToNaturalEnd()`` to scatter the original ``Vec`` into the natural
ordering in a new global ``Vec`` before calling ``VecScatterBegin()``/``VecScatterEnd()``
to scatter the natural ``Vec`` onto all processes.

How Do I Collect To The Zero'th Processor All The Values From A Parallel PETSc ``Vec``
--------------------------------------------------------------------------------------

See FAQ entry on collecting to :ref:`an arbitrary processor <doc_faq_usage_alltoone>`, but
replace

::

   ierr = VecScatterCreateToAll(in_par, &ctx, &out_seq);CHKERRQ(ierr);

with

::

   ierr = VecScatterCreateToZero(in_par, &ctx, &out_seq);CHKERRQ(ierr);

.. note::

   The same ordering considerations as discussed in the aforementioned entry also apply
   here.

How Can I Read In Or Write Out A Sparse Matrix In Matrix Market, Harwell-Boeing, Slapc Or Other ASCII Format?
-------------------------------------------------------------------------------------------------------------

If you can read or write your matrix using Python or MATLAB/Octave, there are
``PetscBinaryIO`` modules for each language that can assist with reading and writing. If you
just want to convert ``MatrixMarket``, you can use:

.. code-block:: console

   > python -m PetscBinaryIO convert matrix.mtx

To produce ``matrix.petsc``. The above assumes you have either installed the module or
added ``$PETSC_DIR/lib/petsc/bin`` to your ``$PYTHONPATH``. You can also call the script
directly or import it from your Python code. There is also a ``PETScBinaryIO.jl`` Julia
package.

For other formats, either adapt one of the above libraries or see the examples in ``$PETSC_DIR/src/mat/tests``, specifically ``ex72.c`` or ``ex78.c``. You will likely need to modify the code slightly to match your required ASCII format.

.. note::

   Never read or write in parallel an ASCII matrix file.

   Instead read in sequentially with a standalone code based on ``ex72.c`` or ``ex78.c``
   then save the matrix with the binary viewer ``PetscViewerBinaryOpen()`` and load the
   matrix in parallel in your "real" PETSc program with ``MatLoad()``.

   For writing save with the binary viewer and then load with the sequential code to store
   it as ASCII.

.. todo:: why? Original entry referenced non-existent (possibly deprecated) functions. Is this still correct?

Does ``TSSetFromOptions()``, ``SNESSetFromOptions()`` or ``KSPSetFromOptions()`` reset all the parameters I previously set or how come my ``TS/SNES/KSPSetXXX()`` does not seem to work?
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. todo:: figure out how to insert links but not break the title link

If ``XXSetFromOptions()`` is used (with ``-xxx_type aaaa``) to change the type of the
object then all parameters associated with the previous type are removed. Otherwise it
does not reset parameters.

``TS/SNES/KSPSetXXX()`` commands that set properties for a particular type of object (such
as ``KSPGMRESSetRestart()``) ONLY work if the object is ALREADY of that type. For example,
with

::

   KSP            ksp;
   PetscErrorCode ierr;

   ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
   ierr = KSPGMRESSetRestart(ksp, 10);CHKERRQ(ierr);

the restart will be ignored since the type has not yet been set to ``KSPGMRES``. To have
those values take effect you should do one of the following:

- Allow setting the type from the command line, if it is not on the command line then the
  default type is automatically set.

::

   /* Create generic object */
   XXXCreate(..,&obj);
   /* Must set all settings here, or default */
   XXXSetFromOptions(obj);

- Hardwire the type in the code, but allow the user to override it via a subsequent
  ``XXXSetFromOptions()`` call. This essentially allows the user to customize what the
  "default" type to of the object.

::

   /* Create generic object */
   XXXCreate(..,&obj);
   /* Set type directly */
   XXXSetYYYYY(obj,...);
   /* Can always change to different type */
   XXXSetFromOptions(obj);

How Do I Compile And Link May Own PETSc Application Codes And Can I Use My Own ``makefile``s Or Rules For Compiling Code, Rather Than PETSc's?
----------------------------------------------------------------------------------------------------------------------------------------------

See the :ref:`section <sec_writing_application_codes>` of the users manual on writing
application codes with PETSc. This provides a simple makefile that can be used to compiler
user code. You are free to modify this makefile or completely replace it with your own
makefile.

Can I Use Cmake To Build My Own Project That Depends On PETSc?
--------------------------------------------------------------

Use `FindPkgConfig.cmake
<https://cmake.org/cmake/help/latest/module/FindPkgConfig.html>`__, which is installed by
default with CMake. PETSc installs ``$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/petsc.pc``,
which can be be read by ``FindPkgConfig.cmake``. If you must use a very old version of
CMake and/or PETSc, you can use the ``FindPETSc.cmake`` module from `this repository
<https://github.com/jedbrown/cmake-modules/>`__.

How Can I Put Carriage Returns In ``PetscPrintf()`` Statements From Fortran?
----------------------------------------------------------------------------

You can use the same notation as in C, just put a ``\n`` in the string. Note that no other C
format instruction is supported.

Or you can use the Fortran concatination ``//`` and ``char(10)``; for example ``'some
string'//char(10)//'another string`` on the next line.
