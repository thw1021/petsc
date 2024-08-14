.. _ch_petscsection:

PetscSection: Connecting Grids to Data
--------------------------------------

The strongest links between solvers and discretizations are

-  the relationship between the layout of data (unknowns) over a mesh (or similar structure) and the data layout in arrays and ``Vec`` used for computation,

-  data (unknowns) partitioning, and

-  ordering of data (unknowns).

To enable modularity, we encode the operations above in simple data
structures that can be understood by the linear algebra (``Vec``, ``Mat``, ``KSP``, ``PC``, ``SNES``), time integrator (``TS``), and optimization (``Tao``) engines in PETSc
without explicit reference to the mesh (topology) or discretization (analysis).

Data Layout by Hand
~~~~~~~~~~~~~~~~~~~

..
  TODO: This text needs additional work so it can be understood without a detailed (or any) understanding of ``DMPLEX`` because the ``PetscSection`` concept is below ``DM`` in the

..
  We may want to even move this introductory ``PetscSection`` material to its own pride of place in the user guide and not inside the ``DMPLEX`` discussion.

Specific entries (or collections of entries) in a ``Vec`` (or a simple array) can be associated with a "location" on a mesh (or other types of data structure) using the ``PetscSection`` object.
A **point** is a ``PetscInt`` that serves as an abstract "index" into arrays from iteratable sets, such as points on a mesh.

``PetscSection`` has two modes of operation.
..
 But you really just mean if there's more than one field...

Mode 1:
^^^^^^^

A ``PetscSection`` associates a set of degrees of freedom (dof), (a small space
:math:`\{e_k\} 0 < k < d_p`), with every point. The number of dof and their meaning may be different for different points. For example, the dof on a cell point may represent pressure
while a dof on a face point may represent velocity. Though points must be
contiguously numbered, they can be in any range
:math:`[\mathrm{pStart}, \mathrm{pEnd})`, which is called a **chart**. A ``PetscSection`` in mode 1 may be thought of as defining a two dimensional array indexed by point in the outer dimension with
a variable length inner dimension indexed by the dof at that point, :math:`v[pStart <= point < pEnd][0 <= dof <d_p]` [#petscsection_footnote]_.

The sequence for constructing a ``PetscSection`` in mode 1 is the following:

#. Specify the range of points, or chart, with ``PetscSectionSetChart()``.

#. Specify the number of dofs per point, with ``PetscSectionSetDof()``. Any values not set will be zero.

#. Set up the ``PetscSection`` with ``PetscSectionSetUp()``.

Below we demonstrate such a process used by ``DMPLEX`` but first we introduce the second mode for working with ``PetscSection``.

Mode 2:
^^^^^^^

A ``PetscSection`` consists of one more **fields** each of which is represented (internally) by a ``PetscSection``.
A ``PetscSection`` in mode 2 may be thought of as defining a three dimensional array indexed by point and field in the outer dimensions with
a variable length inner dimension indexed by the dof at that point. The actual order the values in the array are stored can be set with
``PetscSectionSetPointMajor``\(``PetscSection``\, ``PETSC_TRUE``\, ``PETSC_FALSE``\). In **point major** order all the degrees of freedom for each point for all fields are stored contiguously, otherwise
all degrees of freedom for each field are stored contiguously. With point major order the fields are said to be **interlaced**.

Consider a ``PetscSection`` with 2 fields and 3 points (from 0 to 2) with 1 dof for each point. In point major order the array has the storage
(values for all the fields at point 0, values for all the fields at point 1, values for all the fields at point 2) while in field major order it is
(values for all points in field 0, values for all points in field 1).

The sequence for constructing such a ``PetscSection`` is the following:

#. Specify the range of points, or chart, with ``PetscSectionSetChart()``\. All fields share the same chart.

#. Specify the number of fields with ``PetscSectionSetNumFields()``.

#. Optionally provide a name for the fields with ``PetscSectionSetFieldName()``.

#. Set the number of dof for each point on each field with ``PetscSectionSetFieldDof()``. Again, values not set will be zero.

#. Set the **total** number of dof for each point with ``PetscSectionSetDof()``. Thus value must be greater than or equal to the sum of the values set with
   ``PetscSectionSetFieldDof()`` at that point. Again, values not set will be zero.

#. Set up the ``PetscSection`` with ``PetscSectionSetUp()``.

Once a ``PetscSection`` has been created one can use ``PetscSectionGetStorageSize``\(``PetscSection``\, ``PetscInt`` ``*``) to determine the total number of entries that can be stored in an array or ``Vec``
accessible by the ``PetscSection``. The memory locations in the associated array are found using an **offset** which can be obtained with:

Mode 1:

.. code-block::

   PetscSectionGetOffset(PetscSection, PetscInt point, PetscInt &offset);

Mode 2:

.. code-block::

   PetscSectionGetFieldOffset(PetscSection, PetscInt point, PetscInt field, PetscInt &offset);

The value in the array is then accessed with ``array[offset]``. If there are multiple dof at a point (and field in mode 2) then ``array[offset + 1]``, etc give access to each of those dof.

Using the mesh from
:numref:`fig_doubletMesh`, we provide an example of creating a ``PetscSection`` using mode 1. We can lay out data for
a continuous Galerkin :math:`P_3` finite element method,

.. code-block::

   PetscInt pStart, pEnd, cStart, cEnd, c, vStart, vEnd, v, eStart, eEnd, e;

   DMPlexGetChart(dm, &pStart, &pEnd);
   DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);   // cells
   DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd);   // edges
   DMPlexGetHeightStratum(dm, 2, &vStart, &vEnd);   // vertices, equivalent to DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
   PetscSectionSetChart(s, pStart, pEnd);
   for(c = cStart; c < cEnd; ++c)
       PetscSectionSetDof(s, c, 1);
   for(v = vStart; v < vEnd; ++v)
       PetscSectionSetDof(s, v, 1);
   for(e = eStart; e < eEnd; ++e)
       PetscSectionSetDof(s, e, 2); // two dof on each edge
   PetscSectionSetUp(s);

``DMPlexGetHeightStratum()`` returns all the points of the requested height
in the DAG. Since this problem is in two dimensions the edges are at
height 1 and the vertices at height 2 (the cells are always at height
0). One can also use ``DMPlexGetDepthStratum()`` to use the depth in the
DAG to select the points. ``DMPlexGetDepth(dm,&depth)`` returns the depth
of the DAG, hence ``DMPlexGetDepthStratum(dm,depth-1-h,)`` returns the
same values as ``DMPlexGetHeightStratum(dm,h,)``.

For :math:`P_3` elements there is one degree of freedom at each vertex, 2 along
each edge (resulting in a total of 4 degrees of freedom along each edge
including the vertices, thus being able to reproduce a cubic function)
and 1 degree of freedom within the cell (the bubble function which is
zero along all edges).

Now a PETSc local vector can be created manually using this layout,

.. code-block::

   PetscSectionGetStorageSize(s, &n);
   VecSetSizes(localVec, n, PETSC_DETERMINE);
   VecSetFromOptions(localVec);

When working with ``DMPLEX`` and ``PetscFE`` (see below) one can simply get the sections (and related vectors) with

.. code-block::

   DMSetLocalSection(dm, s);
   DMGetLocalVector(dm, &localVec);
   DMGetGlobalVector(dm, &globalVec);

..
  TODO: This text needs additional work explaining the "constrained dof" business.

A global vector is missing both the shared dofs which are not owned by this process, as well as *constrained* dofs. These constraints represent essential (Dirichlet)
boundary conditions. They are dofs that have a given fixed value, so they are present in local vectors for assembly purposes, but absent
from global vectors since they are never solved for during algebraic solves.

We can indicate constraints in a local section using ``PetscSectionSetConstraintDof()``, to set the number of constrained dofs for a given point, and ``PetscSectionSetConstraintIndices()`` which indicates which dofs on the given point are constrained. Once we have this information, a global section can be created using ``PetscSectionCreateGlobalSection()``, and this is done automatically by the ``DM``. A global section returns :math:`-(dof+1)` for the number of dofs on an unowned point, and :math:`-(off+1)` for its offset on the owning process. This can be used to create global vectors, just as the local section is used to create local vectors.

..
  TODO: This text needs additional work introducing the concept of *fields* in ``PetscSection``. It is unfair to users to not introduce it immediately with ``PetscSection`` since they are ubiquitous.

Data Layout using DMPLEX and PetscFE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``DM`` can automatically create the local section if given a description of the discretization, for example using a ``PetscFE`` object. We demonstrate this by creating
a ``PetscFE`` that can be configured from the command line. It is a single, scalar field, and is added to the ``DM`` using ``DMSetField()``.
When a local or global vector is requested, the ``DM`` builds the local and global sections automatically.

.. code-block::

  DMPlexIsSimplex(dm, &simplex);
  PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe);
  DMSetField(dm, 0, NULL, (PetscObject) fe);
  DMCreateDS(dm);

Here the call to ``DMSetField()`` declares the discretization will have one field with the integer label 0 that has one degree of freedom at each point on the ``DMPlex``.
To get the :math:`P_3` section above, we can either give the option ``-petscspace_degree 3``, or call ``PetscFECreateLagrange()`` and set the degree directly.

Partitioning and Ordering
~~~~~~~~~~~~~~~~~~~~~~~~~

In the same way as ``MatPartitioning`` or
``MatGetOrdering()``, give the results of a partitioning or ordering of a graph defined by a sparse matrix,
``PetscPartitionerDMPlexPartition`` or ``DMPlexPermute`` are encoded in
an ``IS``. However, the graph is not the adjacency graph of the matrix
but the mesh itself. Once the mesh is partitioned and
reordered, the data layout from a ``PetscSection`` can be used to
automatically derive a problem partitioning/ordering.

.. rubric:: Footnotes

.. [#petscsection_footnote] A ``PetscSection`` can be thought of as a generalization of ``PetscLayout``, in the same way that a fiber bundle is a generalization
   of the normal Euclidean basis used in linear algebra. With ``PetscLayout``, we associate a unit vector (:math:`e_i`) with every
   point in the space, and just divide up points between processes.

.. bibliography:: /petsc.bib
    :filter: docname in docnames
