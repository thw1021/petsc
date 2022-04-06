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

- Add ``PETSC_ATTRIBUTE_FORMAT`` to enable compile-time ``printf()``-style format specifier checking and apply it any PETSc functions taking a format string
- Deprecate the use of ``%D`` for printing ``PetscInt`` in favor of ``%" PetscInt_FMT "``. Compilers may emit warnings which users can suppress by defining ``PETSC_SKIP_ATTRIBUTE_FORMAT`` prior to all PETSc header-file inclusions; though they are encouraged to switch to the new style instead.

.. rubric:: Configure/Build:

.. rubric:: Sys:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

- Add ``PetscSectionCreateSubdomainSection()``

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSSetTimeSpan()``, ``TSGetTimeSpan()`` and ``TSGetTimeSpanSolutions()`` to support time span

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
