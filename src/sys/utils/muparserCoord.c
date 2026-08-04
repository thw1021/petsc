#include <petsc/private/petscsysmuparserimpl.h>

/*@
  PetscMuParserCoordFuncCreate - Create a muParser coordinate function from the given expression.

  Not Collective

  Input Parameter:
. expr - the text expression

  Output Parameter:
. cfunc - the created `PetscMuParserCoordFunc`

  Level: intermediate

.seealso: `PetscMuParserCoordFuncDestroy()`
@*/
PetscErrorCode PetscMuParserCoordFuncCreate(const char expr[], PetscMuParserCoordFunc *cfunc)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(cfunc));
#if PetscDefined(HAVE_MUPARSER)
  (*cfunc)->parser = mupCreate(muBASETYPE_FLOAT);
  PetscCheck((*cfunc)->parser, PETSC_COMM_SELF, PETSC_ERR_LIB, "Failed to create muParser");

  mupDefineVar((*cfunc)->parser, "x", &(*cfunc)->x[0]);
  mupDefineVar((*cfunc)->parser, "y", &(*cfunc)->x[1]);
  mupDefineVar((*cfunc)->parser, "z", &(*cfunc)->x[2]);
  // Technically we should support muChar_t here
  mupSetExpr((*cfunc)->parser, expr);
  // Test expression
  PETSC_UNUSED PetscScalar x = mupEval((*cfunc)->parser);
  PetscCheck(!mupError((*cfunc)->parser), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "muParser error in expression \"%s\": %s", expr, mupGetErrorMsg((*cfunc)->parser));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscMuParserCoordFuncDestroy - Destroys a muParser coordinate function.

  Not Collective

  Input Parameter:
. cfunc - the `PetscMuParserCoordFunc` to destroy

  Level: intermediate

.seealso: `PetscMuParserCoordFuncCreate()`
@*/
PetscErrorCode PetscMuParserCoordFuncDestroy(PetscMuParserCoordFunc *cfunc)
{
  PetscFunctionBegin;
  if (!*cfunc) PetscFunctionReturn(PETSC_SUCCESS);
#if PetscDefined(HAVE_MUPARSER)
  if ((*cfunc)->parser) {
    mupRelease((*cfunc)->parser);
    (*cfunc)->parser = NULL;
  }
#endif
  PetscCall(PetscFree(*cfunc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
