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

- Add ``VecNestGetSubVecsRead()`` and ``VecNestRestoreSubVecsRead()`` for read-only access to subvectors
- Add ``VecPointwiseSign()`` and ``VecSignMode``

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Add ``MatConstantDiagonalGetConstant()``

.. rubric:: MatCoarsen:

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

- Add new ``TaoTerm`` object to manipulate objective function terms with many methods
- Add ``TaoComputeHessianSingle()`` convenience function for when the user's code does not compute a preconditioning matrix

.. rubric:: TaoTerm:

- Add ``TAOTERMCALLBACKS`` implementation of ``TaoTerm`` for constructing a term from the callbacks passed to a ``Tao`` object
- Add ``TAOTERMBRGNREGULARIZER`` implementation of ``TaoTerm`` for constructing a term from the callbacks passed to a ``TaoBRGNSetReguarizerObjectiveAndGradientRoutine()``
- Add ``TAOTERMADMMREGULARIZER`` implementation of ``TaoTerm`` for constructing a term from the callbacks passed to a ``TaoADMMSetReguarizerObjectiveAndGradientRoutine()``
- Add ``TAOTERMADMMISFIT`` implementation of ``TaoTerm`` for constructing a term from the callbacks passed to a ``TaoADMMSetMisfitObjectiveAndGradientRoutine()``
- Add ``TAOTERMSHELL`` implementation of ``TaoTerm`` for user-defined callbacks
- Add ``TAOTERMSUM`` implementation of ``TaoTerm`` for scaled, mapped sums of terms
- Add ``TAOTERMHALFL2SQUARED`` implementation of ``TaoTerm`` for a typical squared-norm penalty function
- Add ``TAOTERML1`` implementation of ``TaoTerm`` for a typical 1-norm penalty function

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
