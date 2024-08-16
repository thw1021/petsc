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

While ``PetscSection`` is currently only employed for ``DMPlex``, ``DMForest`` and ``DMNetwork`` mesh descriptions, much of it's operation is general enough to be utilized for other types of discretizations.
This section will explain the basic concepts of a ``PetscSection`` that are generalizable to other mesh descriptions.

.. _sec_petscsection_concept:

General concept
~~~~~~~~~~~~~~~

Specific entries (or collections of entries) in a ``Vec`` (or a simple array) can be associated with a "location" on a mesh (or other types of data structure) using the ``PetscSection`` object.
A **point** is a ``PetscInt`` that serves as an abstract "index" into arrays from iterable sets, such as k-cells in a mesh.
Other iterable set examples can be as simple as the points of a finite difference grid, or cells of a finite volume grid, or as complex as the topological entities of an unstructured mesh (cells, faces, edges, and vertices).

At it's most basic, a ``PetscSection`` is a mapping between the mesh points and a tuple ``(ndof, offset)``, where ``ndof`` is the number of values stored at that mesh point and ``offset`` is the location in the array of that data.
So given the tuple for a mesh point, its data can be accessed by ``array[offset + d]``, where ``d`` in ``[0, ndof)`` is the dof to access.

Charts: Defining mesh points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mesh points for a ``PetscSection`` must be contiguously numbered and are defined to be in some range :math:`[\mathrm{pStart}, \mathrm{pEnd})`, which is called a **chart**.
The chart of a ``PetscSection`` is set via ``PetscSectionSetChart()``.
Note that even though the mesh points must be contiguously numbered, the indexes into the array (defined by the ``(ndof, offset)`` tuple) associated with the ``PetscSection`` need not be.
In other words, there may be elements in the array that are not associated with any mesh points, though this is not often the case.

Defining the (ndof, offset) tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defining the ``(ndof, offset)`` tuple for each mesh point generally first starts with setting the ``ndof`` for each point, which is done using ``PetscSectionSetDof()``.
.. This associates a set of degrees of freedom (dof), (a small space :math:`\{e_k\} 0 < k < ndof`), with every point. 
If ``ndof`` is not set for a mesh point, it is assumed to be 0.

The offset for each mesh point is usually set automatically by ``PetscSectionSetUp()``.
This will concatenate each mesh point's dofs together in the order of the mesh points.
This concatenation can be done in a different order by setting a permutation, which is described in :any:`sec_petscsection_permutation`.

Alternatively, the offset for each mesh point can be set manually by ``PetscSectionSetOffset()``, though this is not commonly needed.

Once the tuples are created, the ``PetscSection`` is ready to use.

Basic Setup Example
^^^^^^^^^^^^^^^^^^^
To summarize, the sequence for constructing a basic ``PetscSection`` is the following:

#. Specify the range of points, or chart, with ``PetscSectionSetChart()``.

#. Specify the number of dofs per point, with ``PetscSectionSetDof()``. Any values not set will be zero.

#. Set up the ``PetscSection`` with ``PetscSectionSetUp()``.

Multiple Fields
~~~~~~~~~~~~~~~

In many discretizations, it is useful to differentiate between different kinds of dofs present on a mesh.
For example, a dof attached to a cell point might represent pressure while dofs on vertices might represent velocity or displacement.
A ``PetscSection`` can represent this additional structure with what are called **fields**.
**Fields** are indexed contiguously from ``[0, num_fields)``.
To set the number of fields for a ``PetscSection``, call ``PetscSectionSetNumFields()``.

Internally, each field is stored in a separate ``PetscSection``.
In fact, all the concepts and functions presented in :any:`sec_petscsection_concept` were actually applied onto the **default field**, which is indexed as ``0``.
The fields inherit the same chart as the "parent" ``PetscSection``.

Setting Up Multiple Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup for a ``PetscSection`` with multiple fields is nearly identical to setup for a single field.

The sequence for constructing such a ``PetscSection`` is the following:

#. Specify the range of points, or chart, with ``PetscSectionSetChart()``\. All fields share the same chart.

#. Specify the number of fields with ``PetscSectionSetNumFields()``.

#. Set the number of dof for each point on each field with ``PetscSectionSetFieldDof()``. Again, values not set will be zero.

#. Set the **total** number of dof for each point with ``PetscSectionSetDof()``. Thus value must be greater than or equal to the sum of the values set with
   ``PetscSectionSetFieldDof()`` at that point. Again, values not set will be zero.

#. Set up the ``PetscSection`` with ``PetscSectionSetUp()``.

Point Major or Field Major
^^^^^^^^^^^^^^^^^^^^^^^^^^
A ``PetscSection`` with one field and and offsets set in ``PetscSectionSetUp()`` may be thought of as defining a two dimensional array indexed by point in the outer dimension with a variable length inner dimension indexed by the dof at that point, :math:`v[\mathrm{pStart} <= point < \mathrm{pEnd}][0 <= dof < \mathrm{ndof}]` [#petscsection_footnote]_.

With multiple fields, this array is now three dimensional, with the outer dimenions being both indexed by mesh points and field points.
Thus, there is a choice on whether to index by points first, or by fields.
In other words, will the array be laid out in a point-major fashion, or field-major.

Point-major ordering corresponds to :math:`v[\mathrm{pStart} <= point < \mathrm{pEnd}][0 <= field < \mathrm{num\_fields}][0 <= dof < \mathrm{ndof}]`.
The all the dofs for each mesh point are stored contiguously, meaning the fields are **interlaced**.
Field-major ordering corresponds to :math:`v[0 <= field < \mathrm{num\_fields}][\mathrm{pStart} <= point < \mathrm{pEnd}][0 <= dof < \mathrm{ndof}]`.
The all the dofs for each field are stored contiguously, meaning the points are **interlaced**.


Consider a ``PetscSection`` with 2 fields and 2 points (from 0 to 2). Let the 0th field have ``ndof=1`` for each point and the 1st field have ``ndof=2`` for each point. 
Denote each array entry :math:`(p_i, f_i, d_i)` for :math:`p_i` being the ith point, :math:`f_i` being the ith field, and :math:`d_i` being the ith dof.

Point-major order would result in:

.. math:: [(p_0, f_0, d_0), (p_0, f_1, d_0), (p_0, f_1, d_1),\\ (p_1, f_0, d_0), (p_1, f_1, d_0), (p_1, f_1, d_1)]

Conversely, field-major ordering would result in:

.. math:: [(p_0, f_0, d_0), (p_1, f_0, d_0),\\ (p_0, f_1, d_0), (p_0, f_1, d_1), (p_1, f_1, d_0), (p_1, f_1, d_1)]

Note that dofs are always contiguous, regardless of the outer dimensional ordering.

Setting the which ordering is done with ``PetscSectionSetPointMajor()``, where ``PETSC_TRUE`` sets point-major and ``PETSC_FALSE`` sets field major.
The current default is for point-major, and many operations on ``DMPlex`` will only work with this ordering. Field-major ordering is provided mainly for compatibility with external packages, such as LibMesh.


Working with data
~~~~~~~~~~~~~~~~~

Once a ``PetscSection`` has been created one can use ``PetscSectionGetStorageSize()`` to determine the total number of entries that can be stored in an array or ``Vec`` accessible by the ``PetscSection``.
This is most often used when creating a new ``Vec`` for a ``PetscSection`` such as:

.. code-block::

   PetscSectionGetStorageSize(s, &n);
   VecSetSizes(localVec, n, PETSC_DETERMINE);
   VecSetFromOptions(localVec);

The memory locations in the associated array are found using an **offset** which can be obtained with:

Single-field ``PetscSection``:

.. code-block::

   PetscSectionGetOffset(PetscSection, PetscInt point, PetscInt &offset);

Multi-field ``PetscSection``:

.. code-block::

   PetscSectionGetFieldOffset(PetscSection, PetscInt point, PetscInt field, PetscInt &offset);

The value in the array is then accessed with ``array[offset + d]``, where ``d`` in ``[0, ndof)`` is the dof to access.


Global Sections: Constrained and Distributed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..
  TODO: This text needs additional work explaining the "constrained dof" business.

A global vector is missing both the ghosted dofs, which are not owned by this process but are stored in the global vector on a different process and *constrained* dofs. These constraints usually represent essential (Dirichlet)
boundary conditions, or algebraic constraints. They are dofs that have a given fixed value, so they are present in local vectors for assembly purposes, but absent
from global vectors since they are not unknowns in the algebraic solves.

We can indicate constraints in a local section using ``PetscSectionSetConstraintDof()``, to set the number of constrained dofs for a given point, and ``PetscSectionSetConstraintIndices()`` which indicates which dofs on the given point are constrained. Once we have this information, a global section can be created using ``PetscSectionCreateGlobalSection()``. This is done automatically by the ``DM``. A global section returns :math:`-(dof+1)` for the number of dofs on an unowned (ghost) point, and :math:`-(off+1)` for its offset on the owning process. This can be used to create global vectors, just as the local section is used to create local vectors.

.. _sec_petscsection_permutation:

Permutation: Changing the order of array data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, when ``PetscSectionSetUp()`` is called, the data laid out in the associated array is assumed to be in the same order of the grid points.
For example, the DoFs associated with grid point 0 appear directly before grid point 1, which appears before grid point 2, etc.

It may be desired to have a different the ordering of data in the array than the order of grid points defined by a section.
For example, one may want grid points associated with the boundary of a domain to appear before points associated with the interior of the domain.

This can be accomplished by either changing the indexes of the grid points themselves, or by informing the section of the change in array ordering.
Either method uses an ``IS`` to define the permutation.

To change the indices of the grid points, call ``PetscSectionPermute()`` to generate a new ``PetscSection`` with the desired grid point permutation.

To just change the array layout without changing the grid point indexing, call ``PetscSectionSetPermutation()``.
This must be called before ``PetscSectionSetUp()`` and will only affect the calculation of the offsets for each grid point.

DMPlex Specific Functionality: Obtaining data from the array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A vanilla ``PetscSection`` gives a relatively naive perspective on the underlying data; it doesn't describe how DoFs attached to a single grid point are ordered or how different grid points relate to each other.
This is where **closures**, **symmetries**, and **closure permutations** come into play.
These features currently target ``DMPlex`` and other unstructured grid descriptions.
A description of those features will be left to :any:`ch_unstructured`.

.. rubric:: Footnotes

.. [#petscsection_footnote] A ``PetscSection`` can be thought of as a generalization of ``PetscLayout``, in the same way that a fiber bundle is a generalization
   of the normal Euclidean basis used in linear algebra. With ``PetscLayout``, we associate a unit vector (:math:`e_i`) with every
   point in the space, and just divide up points between processes.

.. bibliography:: /petsc.bib
    :filter: docname in docnames
