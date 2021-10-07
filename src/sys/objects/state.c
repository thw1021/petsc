
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h>  /*I   "petscsys.h"    I*/

/*@C
   PetscObjectStateGet - Gets the state of any PetscObject,
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example,
         PetscObjectStateGet((PetscObject)mat,&state);

   Output Parameter:
.  state - the object state

   Notes:
    object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   Level: advanced

   seealso: PetscObjectStateIncrease(), PetscObjectStateSet()

@*/
PetscErrorCode PetscObjectStateGet(PetscObject obj,PetscObjectState *state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidIntPointer(state,2);
  *state = obj->state;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectStateSet - Sets the state of any PetscObject,
   regardless of the type.

   Logically Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example,
         PetscObjectStateSet((PetscObject)mat,state);
-  state - the object state

   Notes:
    This function should be used with extreme caution. There is
   essentially only one use for it: if the user calls Mat(Vec)GetRow(Array),
   which increases the state, but does not alter the data, then this
   routine can be used to reset the state.  Such a reset must be collective.

   Level: advanced

   seealso: PetscObjectStateGet(),PetscObjectStateIncrease()

@*/
PetscErrorCode PetscObjectStateSet(PetscObject obj,PetscObjectState state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->state = state;
  PetscFunctionReturn(0);
}

PetscInt PetscObjectComposedDataMax = 10;

#define PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE__(Type,type,PetscType) \
  PetscErrorCode PetscObjectComposedDataIncrease##Type(PetscObject obj) \
  {                                                                     \
    const PetscInt   new_n   = PetscObjectComposedDataMax;              \
    const PetscInt   n       = obj->type##_idmax;                       \
    PetscType        *ar     = obj->type##composeddata;                 \
    PetscType        *new_ar = PETSC_NULLPTR;                           \
    PetscObjectState *ir     = obj->type##composedstate;                \
    PetscObjectState *new_ir = PETSC_NULLPTR;                           \
    PetscErrorCode   ierr;                                              \
                                                                        \
    PetscFunctionBegin;                                                 \
    ierr = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);     \
    ierr = PetscMemcpy(new_ar,ar,n*sizeof(*new_ar));CHKERRQ(ierr);      \
    ierr = PetscMemcpy(new_ir,ir,n*sizeof(*new_ir));CHKERRQ(ierr);      \
    ierr = PetscFree2(ar,ir);CHKERRQ(ierr);                             \
    obj->type##_idmax        = new_n;                                   \
    obj->type##composeddata  = new_ar;                                  \
    obj->type##composedstate = new_ir;                                  \
    PetscFunctionReturn(0);                                             \
  }

#define PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE_(Type,type)            \
  PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE__(Type,type,Petsc##Type)     \
  PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE__(Type##star,type##star,Petsc##Type*)

#define PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE(Type,type)             \
  PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE_(Type,type)

PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE(Int,int)
PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE(Real,real)
PETSC_OBJECT_COMPOSED_DATA_INCREASE_TYPE(Scalar,scalar)

/*@C
   PetscObjectComposedDataRegister - Get an available id for composed data

   Not Collective

   Output parameter:
.  id - an identifier under which data can be stored

   Level: developer

   Notes:
    You must keep this value (for example in a global variable) in order to attach the data to an object or
          access in an object.

   seealso: PetscObjectComposedDataSetInt()

@*/
PetscErrorCode PetscObjectComposedDataRegister(PetscInt *id)
{
  static PetscInt globalcurrentstate = 0;

  PetscFunctionBegin;
  *id = globalcurrentstate++;
  if (globalcurrentstate > PetscObjectComposedDataMax) PetscObjectComposedDataMax += 10;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectGetId - get unique object ID

   Not Collective

   Input Parameter:
.  obj - object

   Output Parameter:
.  id - integer ID

   Level: developer

   Notes:
   The object ID may be different on different processes, but object IDs are never reused so local equality implies global equality.

.seealso: PetscObjectStateGet(), PetscObjectCompareId()
@*/
PetscErrorCode PetscObjectGetId(PetscObject obj,PetscObjectId *id)
{
  PetscFunctionBegin;
  *id = obj->id;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectCompareId - compares the objects ID with a given id

   Not Collective

   Input Parameters:
+  obj - object
-  id - integer ID

   Output Parameter;
.  eq - the ids are equal

   Level: developer

   Notes:
   The object ID may be different on different processes, but object IDs are never reused so local equality implies global equality.

.seealso: PetscObjectStateGet(), PetscObjectGetId()
@*/
PetscErrorCode PetscObjectCompareId(PetscObject obj,PetscObjectId id,PetscBool *eq)
{
  PetscFunctionBegin;
  *eq = (id == obj->id) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}
