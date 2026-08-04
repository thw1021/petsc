#include <petscsysmuparser.h>

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

  mupDefineVar((*cfunc)->parser, "x", &(*cfunc)->x[0]);
  mupDefineVar((*cfunc)->parser, "y", &(*cfunc)->x[1]);
  mupDefineVar((*cfunc)->parser, "z", &(*cfunc)->x[2]);
  // Technically we should support muChar_t here
  mupSetExpr((*cfunc)->parser, expr);
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
