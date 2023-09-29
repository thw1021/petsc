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

- Add ``PCMatGetApplyOperation()`` and ``PCMatSetApplyOperation()``
- Add ``PCReduceFailedReason()``
- Add ``PCSetKSPNestLevel()`` and ``PCSetKSPNestLevel()``
- Refactor ``PCMPI`` to be a private system used automatically when ``-mpi_linear_solver_server`` is used. The ``KSP`` and ``PC`` objects that solve the system now inherit any prefix provided
  initially with ``KSPSetPrefix()`` and do not require the previously required ``mpi_`` prefix
- Add option ``-fieldsplit_1_pc_hpddm_schur_precondition`` to use ``PCHPDDM`` on the Schur complements from ``PCFIELDSPLIT``
- Add ``PCGAMGSetAggressiveSquareGraph()`` to use square graph method for aggressive coarsening that was the previous default coarsening approach before release 3.19
- Add ``PCGAMGSetAggressiveMISk()`` to set the number of levels (k) of aggressive MIS-k coarseing (2 is very common)
- Add ``PCGAMGSetMinDegreeOrderingMISk()`` to use a minimum degree ordering for the (greedy) MIS-k algorithm
- Change ``PCGAMGSetUseParallelCoarseGridSolve()`` to ``PCGAMGSetParallelCoarseGridSolve()``
- Add ``PCGAMGSetRecomputeEstEig()`` to set flag to have Chebyshev recompute its eigen estimates (default set to true)
- Add ``PCGAMGSetLowMemoryFilter()`` Use the system ``MatFilter`` graph/matrix filter, without a temporary copy of the graph, otherwise use method that can be faster

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
