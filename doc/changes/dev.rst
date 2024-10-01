====================
Changes: Development
====================

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

.. rubric:: General:

.. rubric:: Configure/Build:

.. rubric:: Sys:

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: MatCoarsen:

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

- Replace the Fortran array ``DMDALocalInfo`` with a derived type whose entries match the C struct
- Change the Fortran ``DMDAGetNeighbors()`` to return a ``PetscMPIInt, pointer :: n(:)`` and add a Fortran ``DMDARestoreNeighbors()``
- Change the Fortran ``DMDAGetOwnershipRanges()`` to return ``PetscInt, pointer :: n(:)`` and add a Fortran ``DMDARestoreOwnershipRanges()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:

- Deprecate all Fortran function names with the suffix F90 with the equivalent function name without the suffix F90. Functions such as `VecGetArray()`
  now take a Fortran pointer as arguments and hence behave like the deprecated `VecGetArrayF90()`.
- Add ``PETSC_NULL_ENUM_XXX`` to be used instead of ``PETSC_NULL_INTEGER`` when a pointer to an XXX ``enum`` is expected in a PETSc function call
- Add ``PETSC_NULL_INTEGER_ARRAY``, ``PETSC_NULL_SCALAR_ARRAY``, and ``PETSC_NULL_REAL_ARRAY`` for use instead of
  ``PETSC_NULL_INTEGER``, ``PETSC_NULL_SCALAR``,  and ``PETSC_NULL_REAL`` when an input array is expected in a PETSc function call but not
  provided by the user.
- Add ``PETSC_NULL_INTEGER_POINTER`` for arguments that return as arrays, for example, ``PetscInt, pointer :: idx(:)`` but not needed by the user.
- Add automatically generated interface definitions for most PETSc functions to detect illegal usage at compile time
- Add ``PetscObjectIsNull()`` for users to check if a PETSc object is ``NULL``
- Change the PETSc Fortran API so that non-array values, ``v``, passed to PETSc routines expecting arrays must be cast with ``[v]`` in the calling sequence
