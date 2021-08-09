====================
 PETSc Clang Linter
====================

Overview
========

To enforce correctness of high level sanity-checks and perform other maintenance, PETSc employs a ``libclang``-based linter. Its implementation can be found at ``$PETSC_DIR/lib/petsc/bin/maint/petscClangLinter.py``. To run the linter across the entire PETSc source tree use

.. code-block:: console

   $ make lint

To see the full options use

.. code-block:: console

   $ make help-lint

To test the linter itself use

.. code-block:: console

   $ make test-lint


Extending the Linter
====================

Registering New PETSc Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You must register new instances of PETSc classes in the ``classIdMap`` in
``$PETSC_DIR/lib/petsc/bin/maint/petscClangLinter.py``, a ``dict`` that expects its
contents to be in the form

.. code-block:: python

   "CaseSensitiveNameOfPrivateStruct *" : "CaseSensitiveNameOfCorrespondingClassId",

For example, the entry for ``DM`` in the ``classIdMap`` is as follows:

.. code-block:: python

   classIdMap = {
     ...
     "_p_DM *" : "DM_CLASSID",
     ...
   }

Adding a New Function/Macro Checker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkers are the high level "driver" functions called by the linter directly when it
encounters an instance of a registered function. The checkers are responsible for deciding
which sub-checks to run for a particular function.

To illustrate a prototypical checker we will use the checker for
``PetscValidHeaderSpecific()`` as an example.

.. code-block:: python

    def checkPetscValidHeaderSpecific(linter,func,parent):
      __doc__="""
      Specific check for PetscValidHeaderSpecific(obj,classid,idx)
      """
      funcArgs   = linter.getArgumentCursors(func)
      parentArgs = linter.getArgumentCursors(parent)

      obj,classid,idx = funcArgs
      checkMatchingClassid(linter,obj,classid)
      checkMatchingArgNum(linter,obj,idx,parentArgs)
      return

   checkFunctionMap = {
     ...
     "PetscValidHeaderSpecific" : checkPetscValidHeaderSpecific,
     ...
   }

In order to be aware of your function/macro you must first perform a few preliminary steps:

#. Register your checker in the ``checkFunctionMap``, a ``dict`` that expects its contents
   to be in the form of

   .. code-block:: python

      "FullNameOfCheckedFunction" : nameOfPythonFunctionToUse

#. If your function is a function-like macro, you must define an actual function prototype
   of the same name in the source code. To facilitate this, the linter does 2 things
   during parsing:

   #. It defines ``PETSC_CLANG_STATIC_ANALYZER`` as a preprocessor macro.

   #. It treats all input files as C++, allowing you to use templates to simulate the type
      erasure of the regular macro.

   In our case this may look like

   ::

      #if !defined(PETSC_CLANG_STATIC_ANALYZER)

      /* the regular macro definition */
      #define PetscValidHeaderSpecific(obj,classid,id)  \
      do {                                              \
        ...                                             \
      } while (0)

      #else

      /* a fake function version, only visibile for the linter */
      template <typename T>
      void PetscValidHeaderSpecific(T,PetscClassId,int);

      #endif

   .. note::

      You do not have to provide a definition of the template version of your function,
      the declaration is sufficient.

You can now implement the custom checking function.

#. Every checker should be named "checkFullNameOfCheckedFunction" and must accept 3
   arguments. In our example this would be:

   .. code-block:: python

      def checkPetscValidHeaderSpecific(linter,func,parent):

   where the arguments are:

   #. ``linter`` -  an instance of a ``PetscLinter``.

   #. ``func`` - an instance of a ``PetscCursor`` that represents the function/macro to be
      checked.

   #. ``parent`` - an instance of a ``PetscCursor`` that represents the function
      containing the call to ``func``.

#. The next line should contain a descriptive ``__doc__`` string specifying the function
   to be checked, as well as assigning names to each function argument:

   .. code-block:: python

      __doc__="""
      Specific check for PetscValidHeaderSpecific(obj,classid,idx)
      """

#. Now one can extract cursors representing the arguments from the function cursor and
   parent cursor as needed. To do so, use the ``getArgumentCursors()`` utility function
   provided by the linter.

   .. code-block:: python

      funcArgs = linter.getArgumentCursors(func)

   This routine returns a ``tuple`` containing the argument cursors as
   ``PetscCursor``'s. One should further unpack these into individual variables. In case
   one of the variables is not needed, one should ignore it during unpacking using the
   ``_`` symbol.

   .. code-block:: python

      obj,classid,idx = funcArgs

   .. important::

      It is highly recommended that one re-use the exact same names for the decomposed
      arguments as those used in the ``__doc__`` string.

#. Finally one can now apply a host of sub-checks to the arguments. In our example, we
   would like to make certain that

   #. The ``PetscClassId`` in ``classid`` matches the expected ``PetscClassId`` for the
      ``PetscObject`` in ``obj``. For example

      ::

         PetscErrorCode PetscParentFunction(Vec v)
         {
           ...
           PetscValidHeaderSpecific(v,MAT_CLASSID,1);
         }

      would be incorrect. ``v`` is a ``Vec`` so we expect ``VEC_CLASSID`` not ``MAT_CLASSID``.

   #. The argument number given in ``idx`` correctly matches the expected argument number
      in the parent function call. For example

      ::

         PetscErrorCode PetscParentFunction(Vec v, Mat m)
         {
           ...
           PetscValidHeaderSpecific(v,VEC_CLASSID,2);
         }

      would be incorrect. We can see that ``v`` is argument #1 in the parent call, not #2.

   For a full list of available sub-checks see
   ``$PETSC_DIR/lib/petsc/bin/maint/petscClangLinter.py``.
