#include <petscsys.h>

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

PetscErrorCode PetscMuParserCoordFuncDestroy(PetscMuParserCoordFunc *cfunc)
{
  PetscFunctionBegin;
  if (!*cfunc) PetscFunctionReturn(PETSC_SUCCESS);
  if ((*cfunc)->parser) {
#if PetscDefined(HAVE_MUPARSER)
    mupRelease((*cfunc)->parser);
#endif
    (*cfunc)->parser = NULL;
  }
  PetscCall(PetscFree(*cfunc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
