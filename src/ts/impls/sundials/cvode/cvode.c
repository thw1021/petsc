#include <petsc/private/petscimpl.h>
#include <petscts.h>

/*@
  TSCVodeSetConstraints - Sets constraints on the solution using `CVodeSetConstraints()`

  Logically Collective

  Input Parameters:
+ ts          - timestepping context
- constraints - vector whose entries indicate the constraints on the solution for that vector entry,
                see `CVodeSetConstraints()` for the meaning of the entries

  Level: intermediate

  Note:
  `constraints` is referenced internally via `PetscObjectReference()`, if you change the vector entries it may later affect the constraints used by CVode

  You can destroy the vector immediately after making this call.

.seealso: [](sec_sundials), `TSCVodeSetOrder()`, `TS`, `TSCVODEADAMS`, `TSCVODEBDF`, `TSCVodeGetConstraints()`
@*/
PetscErrorCode TSCVodeSetConstraints(TS ts, Vec constraints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSCVodeSetConstraints_C", (TS, Vec), (ts, constraints));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSCVodeGetConstraints - Gets constraints set with `TSCVodeSetConstraints()`

  Logically Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. constraints - vector whose entries indicate the constraints on the solution for that vector entry,
                see `CVodeSetConstraints()` for the meaning of the entries

  Level: intermediate

  Note:
  Do not destroy this vector or change its values.

.seealso: [](sec_sundials), `TSCVodeSetOrder()`, `TS`, `TSCVODEADAMS`, `TSCVODEBDF`, `TSCVodeSetConstraints()`
@*/
PetscErrorCode TSCVodeGetConstraints(TS ts, Vec *constraints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseMethod(ts, "TSCVodeGetConstraints_C", (TS, Vec *), (ts, constraints));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSCVodeSetOrder - Set the maximum order of the `TSCVODEADAMS` and `TSCVODEBDF` methods

  Logically Collective

  Input Parameters:
+ ts    - timestepping context
- order - order of the method

  Options Database Key:
. -ts_cvode_order order - select the order

  Level: intermediate

  Notes:
  The maximum order supported for `TSCVODEADAMS` is 12 and for `TSCVODEBDF` it is 5.

  The default maximum order for `TSCVODEADAMS` is 12 and for `TSCVODEBDF` it is 5.

.seealso: [](sec_sundials), `TSCVodeGetOrder()`, `TS`, `TSCVODEADAMS`, `TSCVODEBDF`
@*/
PetscErrorCode TSCVodeSetOrder(TS ts, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ts, order, 2);
  PetscTryMethod(ts, "TSCVodeSetOrder_C", (TS, PetscInt), (ts, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSCVodeGetOrder - Get the maximum order of the `TSCVODEADAMS` and `TSCVODEBDF` methods

  Not Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. order - order of the method

  Level: intermediate

.seealso: [](sec_sundials), `TSCVodeSetOrder()`, `TS`, `TSCVODEADAMS`, `TSCVODEBDF`
@*/
PetscErrorCode TSCVodeGetOrder(TS ts, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(order, 2);
  PetscUseMethod(ts, "TSCVodeGetOrder_C", (TS, PetscInt *), (ts, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}
