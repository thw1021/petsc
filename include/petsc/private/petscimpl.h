
/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(PETSCIMPL_H)
#define PETSCIMPL_H
#include <petscsys.h>

/* These are used internally by PETSc ASCII IO routines*/
#include <stdarg.h>
PETSC_EXTERN PetscErrorCode PetscVFPrintfDefault(FILE*,const char[],va_list);

#if defined(PETSC_HAVE_CLOSURE)
PETSC_EXTERN PetscErrorCode PetscVFPrintfSetClosure(int (^)(const char*));
#endif

/*
   All major PETSc data structures have a common core; this is defined
   below by PETSCHEADER.

   PetscHeaderCreate() should be used whenever creating a PETSc structure.
*/

/*
   PetscOps: structure of core operations that all PETSc objects support.

      getcomm()         - Gets the object's communicator.
      view()            - Is the routine for viewing the entire PETSc object; for
                          example, MatView() is the general matrix viewing routine.
                          This is used by PetscObjectView((PetscObject)obj) to allow
                          viewing any PETSc object.
      destroy()         - Is the routine for destroying the entire PETSc object;
                          for example,MatDestroy() is the general matrix
                          destruction routine.
                          This is used by PetscObjectDestroy((PetscObject*)&obj) to allow
                          destroying any PETSc object.
      compose()         - Associates a PETSc object with another PETSc object with a name
      query()           - Returns a different PETSc object that has been associated
                          with the first object using a name.
      composefunction() - Attaches an a function to a PETSc object with a name.
      queryfunction()   - Requests a registered function that has been attached to a PETSc object.
*/

typedef struct {
   PetscErrorCode (*getcomm)(PetscObject,MPI_Comm *);
   PetscErrorCode (*view)(PetscObject,PetscViewer);
   PetscErrorCode (*destroy)(PetscObject*);
   PetscErrorCode (*compose)(PetscObject,const char[],PetscObject);
   PetscErrorCode (*query)(PetscObject,const char[],PetscObject *);
   PetscErrorCode (*composefunction)(PetscObject,const char[],void (*)(void));
   PetscErrorCode (*queryfunction)(PetscObject,const char[],void (**)(void));
} PetscOps;

typedef enum {PETSC_FORTRAN_CALLBACK_CLASS,PETSC_FORTRAN_CALLBACK_SUBTYPE,PETSC_FORTRAN_CALLBACK_MAXTYPE} PetscFortranCallbackType;
typedef int PetscFortranCallbackId;
#define PETSC_SMALLEST_FORTRAN_CALLBACK ((PetscFortranCallbackId)1000)
PETSC_EXTERN PetscErrorCode PetscFortranCallbackRegister(PetscClassId,const char*,PetscFortranCallbackId*);
PETSC_EXTERN PetscErrorCode PetscFortranCallbackGetSizes(PetscClassId,PetscInt*,PetscInt*);

typedef struct {
  void (*func)(void);
  void *ctx;
} PetscFortranCallback;

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
#define PETSC_MAX_OPTIONS_HANDLER 5
typedef struct _p_PetscObject {
  PetscClassId         classid;
  PetscOps             bops[1];
  MPI_Comm             comm;
  PetscInt             type;
  PetscLogDouble       flops,time,mem,memchildren;
  PetscObjectId        id;
  PetscInt             refct;
  PetscMPIInt          tag;
  PetscFunctionList    qlist;
  PetscObjectList      olist;
  char                 *class_name;    /*  for example, "Vec" */
  char                 *description;
  char                 *mansec;
  char                 *type_name;     /*  this is the subclass, for example VECSEQ which equals "seq" */
  PetscObject          parent;
  PetscObjectId        parentid;
  char*                name;
  char                 *prefix;
  PetscInt             tablevel;
  void                 *cpp;
  PetscObjectState     state;
  PetscInt             int_idmax,        intstar_idmax;
  PetscObjectState     *intcomposedstate,*intstarcomposedstate;
  PetscInt             *intcomposeddata, **intstarcomposeddata;
  PetscInt             real_idmax,        realstar_idmax;
  PetscObjectState     *realcomposedstate,*realstarcomposedstate;
  PetscReal            *realcomposeddata, **realstarcomposeddata;
  PetscInt             scalar_idmax,        scalarstar_idmax;
  PetscObjectState     *scalarcomposedstate,*scalarstarcomposedstate;
  PetscScalar          *scalarcomposeddata, **scalarstarcomposeddata;
  void                 (**fortran_func_pointers)(void);                  /* used by Fortran interface functions to stash user provided Fortran functions */
  PetscInt             num_fortran_func_pointers;                        /* number of Fortran function pointers allocated */
  PetscFortranCallback *fortrancallback[PETSC_FORTRAN_CALLBACK_MAXTYPE];
  PetscInt             num_fortrancallback[PETSC_FORTRAN_CALLBACK_MAXTYPE];
  void                 *python_context;
  PetscErrorCode       (*python_destroy)(void*);

  PetscInt             noptionhandler;
  PetscErrorCode       (*optionhandler[PETSC_MAX_OPTIONS_HANDLER])(PetscOptionItems*,PetscObject,void*);
  PetscErrorCode       (*optiondestroy[PETSC_MAX_OPTIONS_HANDLER])(PetscObject,void*);
  void                 *optionctx[PETSC_MAX_OPTIONS_HANDLER];
  PetscBool            optionsprinted;
#if defined(PETSC_HAVE_SAWS)
  PetscBool            amsmem;          /* if PETSC_TRUE then this object is registered with SAWs and visible to clients */
  PetscBool            amspublishblock; /* if PETSC_TRUE and publishing objects then will block at PetscObjectSAWsBlock() */
#endif
  PetscOptions         options;         /* options database used, NULL means default */
  PetscBool            donotPetscObjectPrintClassNamePrefixType;
} _p_PetscObject;

#define PETSCHEADER(ObjectOps) \
  _p_PetscObject hdr;          \
  ObjectOps      ops[1]

#define  PETSCFREEDHEADER -1

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscObjectDestroyFunction)(PetscObject*); /* force cast in next macro to NEVER use extern "C" style */
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscObjectViewFunction)(PetscObject,PetscViewer);

/*@C
    PetscHeaderCreate - Creates a PETSc object of a particular class

    Input Parameters:
+   classid - the classid associated with this object (for example VEC_CLASSID)
.   class_name - string name of class; should be static (for example "Vec")
.   descr - string containing short description; should be static (for example "Vector")
.   mansec - string indicating section in manual pages; should be static (for example "Vec")
.   comm - the MPI Communicator
.   destroy - the destroy routine for this object (for example VecDestroy())
-   view - the view routine for this object (for example VecView())

    Output Parameter:
.   h - the newly created object

    Level: developer

.seealso: PetscHeaderDestroy(), PetscClassIdRegister()

@*/
#define PetscHeaderCreate(h,classid,class_name,descr,mansec,comm,destroy,view) \
  (PetscNew(&(h)) || \
   PetscHeaderCreate_Private((PetscObject)(h),classid,class_name,descr,mansec,comm,(PetscObjectDestroyFunction)(destroy),(PetscObjectViewFunction)(view)) || \
   PetscLogObjectCreate(h) || \
   PetscLogObjectMemory((PetscObject)(h),sizeof(*(h))))

PETSC_EXTERN PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj);
PETSC_EXTERN PetscErrorCode PetscHeaderCreate_Private(PetscObject,PetscClassId,const char[],const char[],const char[],MPI_Comm,PetscObjectDestroyFunction,PetscObjectViewFunction);

/*@C
    PetscHeaderDestroy - Final step in destroying a PetscObject

    Input Parameters:
.   h - the header created with PetscHeaderCreate()

    Level: developer

.seealso: PetscHeaderCreate()
@*/
#define PetscHeaderDestroy(h) (PetscHeaderDestroy_Private((PetscObject)(*(h))) || PetscFree(*(h)))

PETSC_EXTERN PetscErrorCode PetscHeaderDestroy_Private(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectCopyFortranFunctionPointers(PetscObject,PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSetFortranCallback(PetscObject,PetscFortranCallbackType,PetscFortranCallbackId*,void(*)(void),void *ctx);
PETSC_EXTERN PetscErrorCode PetscObjectGetFortranCallback(PetscObject,PetscFortranCallbackType,PetscFortranCallbackId,void(**)(void),void **ctx);

PETSC_INTERN PetscErrorCode PetscCitationsInitialize(void);
PETSC_INTERN PetscErrorCode PetscFreeMPIResources(void);
PETSC_INTERN PetscErrorCode PetscOptionsHasHelpIntro_Internal(PetscOptions,PetscBool*);


PETSC_EXTERN PetscBool PetscCheckPointer(const void*,PetscDataType);
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscBool PetscCheckMpiGpuAwareness(void);
#endif
/*
    Macros to test if a PETSc object is valid and if pointers are valid
*/
#if !defined(PETSC_USE_DEBUG)

#define PetscValidHeaderSpecific(h,ck,arg) do {(void)(h);} while (0)
#define PetscValidHeaderSpecificType(h,ck,arg,t) do {(void)(h);} while (0)
#define PetscValidHeader(h,arg) do {(void)(h);} while (0)
#define PetscValidPointer(h,arg) do {(void)(h);} while (0)
#define PetscValidCharPointer(h,arg) do {(void)(h);} while (0)
#define PetscValidIntPointer(h,arg) do {(void)(h);} while (0)
#define PetscValidBoolPointer(h,arg) do {(void)(h);} while (0)
#define PetscValidScalarPointer(h,arg) do {(void)(h);} while (0)
#define PetscValidRealPointer(h,arg) do {(void)(h);} while (0)
#define PetscValidFunction(h,arg) do {(void)(h);} while (0)

#else

/*  This check is for subtype methods such as DMDAGetCorners() that do not use the PetscTryMethod() or PetscUseMethod() paradigm */
#define PetscValidHeaderSpecificType(h,ck,arg,t) \
  do {   \
    PetscErrorCode _7_ierr; \
    PetscBool      _7_same; \
    PetscValidHeaderSpecific(h,ck,arg); \
    _7_ierr = PetscObjectTypeCompare((PetscObject)(h),t,&_7_same);CHKERRQ(_7_ierr); \
    if (!_7_same) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong subtype object:Parameter # %d must have implementation %s it is %s",arg,t,((PetscObject)(h))->type_name); \
  } while (0)

#define PetscValidHeaderSpecific(h,ck,arg)                              \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_OBJECT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg); \
    if (((PetscObject)(h))->classid != ck) {                            \
      if (((PetscObject)(h))->classid == PETSCFREEDHEADER) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg); \
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong type of object: Parameter # %d",arg); \
    }                                                                   \
  } while (0)

#define PetscValidHeader(h,arg)                                         \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Object: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_OBJECT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object: Parameter # %d",arg); \
    if (((PetscObject)(h))->classid == PETSCFREEDHEADER) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Object already free: Parameter # %d",arg); \
    else if (((PetscObject)(h))->classid < PETSC_SMALLEST_CLASSID || ((PetscObject)(h))->classid > PETSC_LARGEST_CLASSID) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid type of object: Parameter # %d",arg); \
  } while (0)

#define PetscValidPointer(h,arg)                                        \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_CHAR)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer: Parameter # %d",arg); \
  } while (0)

#define PetscValidCharPointer(h,arg)                                    \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg);\
    if (!PetscCheckPointer(h,PETSC_CHAR)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to char: Parameter # %d",arg); \
  } while (0)

#define PetscValidIntPointer(h,arg)                                     \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_INT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscInt: Parameter # %d",arg); \
  } while (0)

#define PetscValidBoolPointer(h,arg)                                    \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_BOOL)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscBool: Parameter # %d",arg); \
  } while (0)

#define PetscValidScalarPointer(h,arg)                                  \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_SCALAR)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscScalar: Parameter # %d",arg); \
  } while (0)

#define PetscValidRealPointer(h,arg)                                    \
  do {                                                                  \
    if (!(h)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Pointer: Parameter # %d",arg); \
    if (!PetscCheckPointer(h,PETSC_REAL)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Invalid Pointer to PetscReal: Parameter # %d",arg); \
  } while (0)

#define PetscValidFunction(f,arg)                                       \
  do {                                                                  \
    if (!(f)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null Function Pointer: Parameter # %d",arg); \
  } while (0)

#endif

#define PetscSorted(n,idx,sorted)           \
  do {                                      \
    PetscInt _i_;                           \
    (sorted) = PETSC_TRUE;                  \
    for (_i_ = 1; _i_ < (n); _i_++)         \
      if ((idx)[_i_] < (idx)[_i_ - 1])      \
        { (sorted) = PETSC_FALSE; break; }  \
  } while(0)

#if !defined(PETSC_USE_DEBUG)

#define PetscCheckSameType(a,arga,b,argb) do {(void)(a);(void)(b);} while (0)
#define PetscCheckTypeName(a,type) do {(void)(a);} while (0)
#define PetscCheckTypeNames(a,type1,type2) do {(void)(a);} while (0)
#define PetscValidType(a,arg) do {(void)(a);} while (0)
#define PetscCheckSameComm(a,arga,b,argb) do {(void)(a);(void)(b);} while (0)
#define PetscCheckSameTypeAndComm(a,arga,b,argb) do {(void)(a);(void)(b);} while (0)
#define PetscValidLogicalCollectiveScalar(a,b,arg) do {(void)(a);(void)(b);} while (0)
#define PetscValidLogicalCollectiveReal(a,b,arg) do {(void)(a);(void)(b);} while (0)
#define PetscValidLogicalCollectiveInt(a,b,arg) do {(void)(a);(void)(b);} while (0)
#define PetscValidLogicalCollectiveMPIInt(a,b,arg) do {(void)(a);(void)(b);} while (0)
#define PetscValidLogicalCollectiveBool(a,b,arg) do {(void)(a);(void)(b);} while (0)
#define PetscValidLogicalCollectiveEnum(a,b,arg) do {(void)(a);(void)(b);} while (0)
#define PetscCheckSorted(n,idx) do {(void)(n);(void)(idx);} while (0)

#else

/*
    For example, in the dot product between two vectors,
  both vectors must be either Seq or MPI, not one of each
*/
#define PetscCheckSameType(a,arga,b,argb)                               \
  do {                                                                  \
    if (((PetscObject)(a))->type != ((PetscObject)(b))->type) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Objects not of same type: Argument # %d and %d",arga,argb); \
  } while (0)
/*
    Check type_name
*/
#define PetscCheckTypeName(a,type)                                      \
  do {                                                                  \
    PetscBool      _7_match;                                            \
    PetscErrorCode _7_ierr;                                             \
    _7_ierr = PetscObjectTypeCompare(((PetscObject)(a)),(type),&_7_match);CHKERRQ(_7_ierr); \
    if (!_7_match) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Object (%s) is not %s",(char*)(((PetscObject)(a))->type_name),type); \
  } while (0)

#define PetscCheckTypeNames(a,type1,type2)                              \
  do {                                                                  \
    PetscBool      _7_match;                                            \
    PetscErrorCode _7_ierr;                                             \
    _7_ierr = PetscObjectTypeCompareAny(((PetscObject)(a)),&_7_match,(type1),(type2),"");CHKERRQ(_7_ierr); \
    if (!_7_match) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Object (%s) is not %s or %s",(char*)(((PetscObject)(a))->type_name),type1,type2); \
  } while (0)
/*
   Use this macro to check if the type is set
*/
#define PetscValidType(a,arg)                                           \
  do {                                                                  \
    if (!((PetscObject)(a))->type_name) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"%s object's type is not set: Argument # %d",((PetscObject)(a))->class_name,arg); \
  } while (0)
/*
   Sometimes object must live on same communicator to inter-operate
*/
#define PetscCheckSameComm(a,arga,b,argb)                               \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt    _7_flag;                                             \
    _7_ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)(a)),PetscObjectComm((PetscObject)(b)),&_7_flag);CHKERRQ(_7_ierr); \
    if (_7_flag != MPI_CONGRUENT && _7_flag != MPI_IDENT) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the two objects: Argument # %d and %d flag %d",arga,argb,_7_flag); \
  } while (0)

#define PetscCheckSameTypeAndComm(a,arga,b,argb)        \
  do {                                                  \
    PetscCheckSameType(a,arga,b,argb);                  \
    PetscCheckSameComm(a,arga,b,argb);                  \
  } while (0)

#define PetscValidLogicalCollectiveScalar(a,b,arg)                      \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscScalar b0=(b);                                                 \
    PetscReal b1[5],b2[5];                                              \
    if (PetscIsNanScalar(b0)) {b1[4] = 1;} else {b1[4] = 0;};           \
    b1[0] = -PetscRealPart(b0); b1[1] = PetscRealPart(b0); b1[2] = -PetscImaginaryPart(b0); b1[3] = PetscImaginaryPart(b0); \
    _7_ierr = MPI_Allreduce(b1,b2,5,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)(a)));CHKERRQ(_7_ierr); \
    if (!(b2[4] > 0) && !(PetscEqualReal(-b2[0],b2[1]) && PetscEqualReal(-b2[2],b2[3]))) SETERRQ1(PetscObjectComm((PetscObject)(a)),PETSC_ERR_ARG_WRONG,"Scalar value must be same on all processes, argument # %d",arg); \
  } while (0)

#define PetscValidLogicalCollectiveReal(a,b,arg)                        \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscReal b0=(b),b1[3],b2[3];                                       \
    if (PetscIsNanReal(b0)) {b1[2] = 1;} else {b1[2] = 0;};             \
    b1[0] = -b0; b1[1] = b0;                                            \
    _7_ierr = MPI_Allreduce(b1,b2,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)(a)));CHKERRQ(_7_ierr); \
    if (!(b2[2] > 0) && !PetscEqualReal(-b2[0],b2[1])) SETERRQ1(PetscObjectComm((PetscObject)(a)),PETSC_ERR_ARG_WRONG,"Real value must be same on all processes, argument # %d",arg); \
  } while (0)

#define PetscValidLogicalCollectiveInt(a,b,arg)                         \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscInt b0=(b),b1[2],b2[2];                                        \
    b1[0] = -b0; b1[1] = b0;                                            \
    _7_ierr = MPIU_Allreduce(b1,b2,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)(a)));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)(a)),PETSC_ERR_ARG_WRONG,"Int value must be same on all processes, argument # %d",arg); \
  } while (0)

#define PetscValidLogicalCollectiveMPIInt(a,b,arg)                      \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt b0=(b),b1[2],b2[2];                                     \
    b1[0] = -b0; b1[1] = b0;                                            \
    _7_ierr = MPIU_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)(a)));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)(a)),PETSC_ERR_ARG_WRONG,"PetscMPIInt value must be same on all processes, argument # %d",arg); \
  } while (0)

#define PetscValidLogicalCollectiveBool(a,b,arg)                        \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt b0=(PetscMPIInt)(b),b1[2],b2[2];                        \
    b1[0] = -b0; b1[1] = b0;                                            \
    _7_ierr = MPIU_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)(a)));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)(a)),PETSC_ERR_ARG_WRONG,"Bool value must be same on all processes, argument # %d",arg); \
  } while (0)

#define PetscValidLogicalCollectiveEnum(a,b,arg)                        \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscMPIInt b0=(PetscMPIInt)(b),b1[2],b2[2];                        \
    b1[0] = -b0; b1[1] = b0;                                            \
    _7_ierr = MPIU_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)(a)));CHKERRQ(_7_ierr); \
    if (-b2[0] != b2[1]) SETERRQ1(PetscObjectComm((PetscObject)(a)),PETSC_ERR_ARG_WRONG,"Enum value must be same on all processes, argument # %d",arg); \
  } while (0)

#define PetscCheckSorted(n,idx)                                                                   \
  do {                                                                                            \
    PetscBool _1_flg;                                                                             \
    PetscSorted(n,idx,_1_flg);                                                                    \
    if (!_1_flg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input array needs to be sorted"); \
  } while (0)

#endif

#include <string.h>
#define MIN_GALLOP_CONST_GLOBAL 8
static PetscInt MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;

#define BYTESWAP(a,b,t,size)                                            \
  do {                                                                  \
    memcpy((t),(b),(size));                                             \
    memmove((b),(a),(size));                                            \
    memcpy((a),(t),(size));                                             \
  } while (0)

/* Start left look right. Looking for e.g. B[0] in A or mergelo. l inclusive, r inclusive. Returns first m such that arr[m] >
 x. Output also inclusive */
PETSC_STATIC_INLINE PetscErrorCode PetscGallopSearchLeft_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), const PetscInt l, const PetscInt r, const void *x, PetscInt *m)
{
  PetscInt last = l, k = 1, mid, cur = l+1;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(r < l)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"r %D < l %D in PetscGallopSearchLeft",r,l);
  if (PetscUnlikely(!r-l)) {*m = l;PetscFunctionReturn(0);}
  if ((*cmp)(x, arr+l*size) < 0) {*m = l;PetscFunctionReturn(0);}
  while (PETSC_TRUE) {
    if (cur > r) {cur = r; break;}
    if ((*cmp)(x, arr+cur*size) < 0) break;
    last = cur;
    cur += (k <<= 1) + 1; ++k;
  }
  /* standard binary search but take last 0 mid 0 cur 1 into account*/
  while (cur > last + 1) {
    mid = last + ((cur - last) >> 1);
    if ((*cmp)(x, arr+mid*size) < 0) {
      cur = mid;
    } else {
      last = mid;
    }
  }
  *m = cur;
  PetscFunctionReturn(0);
}

/* Start right look left. Looking for e.g. B[-1] in A or mergehi. l inclusive, r inclusive. Returns last m such that arr[m]
 < x. Output also inclusive */
PETSC_STATIC_INLINE PetscErrorCode PetscGallopSearchRight_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), const PetscInt l, const PetscInt r, const void *x, PetscInt *m)
{
  PetscInt last = r, k = 1, mid, cur = r-1;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(r < l)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"r %D < l %D in PetscGallopSearchRight",r,l);
  if (PetscUnlikely(!r-l)) {*m = r;PetscFunctionReturn(0);}
  if ((*cmp)(x, arr+r*size) > 0) {*m = r;PetscFunctionReturn(0);}
  while (PETSC_TRUE) {
    if (cur < l) {cur = l; break;}
    if ((*cmp)(x, arr+cur*size) > 0) break;
    last = cur;
    cur -= (k <<= 1) + 1; ++k;
  }
  /* standard binary search but take last r-1 mid r-1 cur r-2 into account*/
  while (last > cur + 1) {
    mid = last - ((last - cur) >> 1);
    if ((*cmp)(x, arr+mid*size) > 0) {
      cur = mid;
    } else {
      last = mid;
    }
  }
  *m = cur;
  PetscFunctionReturn(0);
}

/* Mergesort where size of left half <= size of right half, so mergesort is done left to right. Arr should be pointer to
 complete array, left is first index of left array, mid is first index of right array, right is last index of right
 array */
PETSC_STATIC_INLINE PetscErrorCode PetscTimSortMergeLo_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), char *tarr, const PetscInt left, const PetscInt mid, const PetscInt right)
{
  PetscInt       i = 0, j = mid, k = left, llen = mid-left, gallopleft = 0, gallopright = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  memcpy(tarr, arr+left*size, llen*size);
  while ((i < llen) && (j <= right)) {
    if ((*cmp)(tarr+i*size, arr+j*size) < 0) {
      memcpy(arr+k*size, tarr+i*size, size);
      ++k; ++i;
      gallopright = 0;
      if (++gallopleft >= MIN_GALLOP_GLOBAL && i < llen) {
        PetscInt l1, l2, diff1, diff2;
        ++MIN_GALLOP_GLOBAL;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search temp for right[j], can move up to that of temp into arr immediately */
          ierr = PetscGallopSearchLeft_Private(tarr, size, cmp, i, llen-1, arr+j*size, &l1);CHKERRQ(ierr);
          diff1 = l1-i;
          memcpy(arr+k*size, tarr+i*size, diff1*size);
          k += diff1;
          i = l1;
          /* search right for temp[i], can move up to that many of right into arr */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, j, right, tarr+i*size, &l2);CHKERRQ(ierr);
          diff2 = l2-j;
          memmove(arr+k*size, arr+j*size, diff2*size);
          k += diff2;
          j = l2;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    } else {
      memmove(arr+k*size, arr+j*size, size);
      ++k; ++j;
      gallopleft = 0;
      if (++gallopright >= MIN_GALLOP_GLOBAL && j <= right) {
        PetscInt l1, l2, diff1, diff2;
        ++MIN_GALLOP_GLOBAL;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search right for temp[i], can move up to that many of right into arr */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, j, right, tarr+i*size, &l2);CHKERRQ(ierr);
          diff2 = l2-j;
          memmove(arr+k*size, arr+j*size, diff2*size);
          k += diff2;
          j = l2;
          /* search temp for right[j], can copy up to that of temp into arr immediately */
          ierr = PetscGallopSearchLeft_Private(tarr, size, cmp, i, llen-1, arr+j*size, &l1);CHKERRQ(ierr);
          diff1 = l1-i;
          memcpy(arr+k*size, tarr+i*size, diff1*size);
          k += diff1;
          i = l1;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff1 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    }
  }
  if (i<llen) {memcpy(arr+k*size, tarr+i*size, (llen-i)*size);}
  PetscFunctionReturn(0);
}

/* Mergesort where size of right half < size of left half, so mergesort is done right to left. Arr should be pointer to
 complete array, left is first index of left array, mid is first index of right array, right is last index of right
 array */
PETSC_STATIC_INLINE PetscErrorCode PetscTimSortMergeHi_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), char *tarr, const PetscInt left, const PetscInt mid, const PetscInt right)
{
  PetscInt       i = right-mid, j = mid-1, k = right, rlen = right-mid+1, gallopleft = 0, gallopright = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  memcpy(tarr, arr+mid*size, rlen*size);
  while ((i >= 0) && (j >= left)) {
    if ((*cmp)(tarr+i*size, arr+j*size) > 0) {
      memcpy(arr+k*size, tarr+i*size, size);
      --k; --i;
      gallopleft = 0;
      if (++gallopright >= MIN_GALLOP_GLOBAL && i >= 0) {
        PetscInt l1, l2, diff1, diff2;
        ++MIN_GALLOP_GLOBAL;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search temp for left[j], can copy up to that many of temp into arr */
          ierr = PetscGallopSearchRight_Private(tarr, size, cmp, 0, i, arr+j*size, &l1);CHKERRQ(ierr);
          diff1 = i-l1;
          memcpy(arr+(k-diff1+1)*size, tarr+(l1+1)*size, diff1*size);
          k -= diff1;
          i = l1;
          /* search left for temp[i], can move up to that many of left up arr */
          ierr = PetscGallopSearchRight_Private(arr, size, cmp, left, j, tarr+i*size, &l2);CHKERRQ(ierr);
          diff2 = j-l2;
          memmove(arr+(k-diff2+1)*size, arr+(l2+1)*size, diff2*size);
          k -= diff2;
          j = l2;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    } else {
      memmove(arr+k*size, arr+j*size, size);
      --k; --j;
      gallopright = 0;
      if (++gallopleft >= MIN_GALLOP_GLOBAL && j >= left) {
        PetscInt l1, l2, diff1, diff2;
        ++MIN_GALLOP_GLOBAL;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search left for temp[i], can move up to that many of left up arr */
          ierr = PetscGallopSearchRight_Private(arr, size, cmp, left, j, tarr+i*size, &l2);CHKERRQ(ierr);
          diff2 = j-l2;
          memmove(arr+(k-diff2+1)*size, arr+(l2+1)*size, diff2*size);
          k -= diff2;
          j = l2;
          /* search temp for left[j], can copy up to that many of temp into arr */
          ierr = PetscGallopSearchRight_Private(tarr, size, cmp, 0, i, arr+j*size, &l1);CHKERRQ(ierr);
          diff1 = i-l1;
          memcpy(arr+(k-diff1+1)*size, tarr+(l1+1)*size, diff1*size);
          k -= diff1;
          i = l1;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    }
  }
  if (i >= 0) {memcpy(arr+left*size, tarr, (i+1)*size);}
  PetscFunctionReturn(0);
}

/* Left is inclusive lower bound of array slice, start is start location of unsorted section, right is inclusive upper
 bound of array slice. If unsure of where unsorted section starts or if entire length is unsorted pass start = left */
PETSC_STATIC_INLINE PetscErrorCode PetscInsertionSort_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), const PetscInt left, const PetscInt start, const PetscInt right)
{
  PetscInt       i = start;
  char           *t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(size+size, &t);CHKERRQ(ierr);
  if (start == left) ++i;
  for (; i <= right; ++i) {
    PetscInt j = i-1;
    memcpy(t, arr+i*size, size);
    while ((j >= left) && ((*cmp)(t, arr+j*size) < 0)) {
      BYTESWAP(arr+(j+1)*size,arr+j*size,t+size,size);
      --j;
    }
    memcpy(arr+(j+1)*size, t, size);
  }
  ierr = PetscFree(t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* See PetscInsertionSort_Private */
PETSC_STATIC_INLINE PetscErrorCode PetscBinaryInsertionSort_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), const PetscInt left, const PetscInt start, const PetscInt right)
{
  PetscInt       i = start;
  char           *t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(size+size, &t);CHKERRQ(ierr);
  if (start == left) ++i;
  for (; i <= right; ++i) {
    PetscInt l = left, r = i, j;
    memcpy(t, arr+i*size, size);
    do {
      PetscInt m;
      m = l + ((r - l) >> 1);
      if ((*cmp)(t, arr+m*size) < 0) {
        r = m;
      } else {
        l = m + 1;
      }
    } while (l < r);
    for (j = i; j > l; --j) {BYTESWAP(arr+j*size,arr+(j-1)*size,t+size,size);}
    memcpy(arr+l*size, t, size);
  }
  ierr = PetscFree(t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt size;
  PetscInt start;
} PetscTimSortStack;

typedef struct {
  size_t  size;
  size_t  maxsize;
  char    *ptr;
} PetscTimSortBuffer;

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortResizeBuffer_Private(PetscTimSortBuffer *buff, const size_t size, const PetscInt newSize)
{
  PetscFunctionBegin;
  if (PetscLikely(newSize*size <= buff->size)) PetscFunctionReturn(0);
  {
    /* Can't be larger than n, there is merit to simply allocating buff to n to begin with */
    PetscErrorCode ierr;
    size_t         newMax = PetscMin((newSize*newSize)*size, buff->maxsize);
    ierr = PetscFree(buff->ptr);CHKERRQ(ierr);
    ierr = PetscMalloc(newMax, &buff->ptr);CHKERRQ(ierr);
    buff->size = newMax;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortForceCollapse_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), PetscTimSortBuffer *buff, PetscTimSortStack *stack, PetscInt stacksize)
{
  PetscFunctionBegin;
  while (stacksize) {
    PetscInt       l, m = stack[stacksize].start, r;
    PetscErrorCode ierr;

    /* A = stack[i-1], B = stack[i] */
    /* Search A for B[0] insertion */
    ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[stacksize-1].start, stack[stacksize].start-1, arr+(stack[stacksize].start)*size, &l);CHKERRQ(ierr);
    /* l == m-1 means sorted */
    if (l < m-1) {
      /* Search B for A[-1] insertion */
      ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[stacksize].start, stack[stacksize].start+stack[stacksize].size-1, arr+(stack[stacksize].start-1)*size, &r);CHKERRQ(ierr);
      if (m-l <= r-m) {
        ierr = PetscTimSortResizeBuffer_Private(buff, size, m-l+1);CHKERRQ(ierr);
        ierr = PetscTimSortMergeLo_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
      } else {
        ierr = PetscTimSortResizeBuffer_Private(buff, size, r-m+1);CHKERRQ(ierr);
        ierr = PetscTimSortMergeHi_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
      }
    }
    /* Update A with merge */
    stack[stacksize-1].size += stack[stacksize].size;
    --stacksize;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortMergeCollapse_Private(char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), PetscTimSortBuffer *buff, PetscTimSortStack *stack, PetscInt *stacksize)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  i = *stacksize;
  while (i) {
    PetscInt l, m, r, itemp = i;

    if (i == 1) {
      /* A = stack[i-1], B = stack[i] */
      if (stack[i-1].size < stack[i].size) {
        m = stack[i].start;
        /* Search A for B[0] insertion */
        ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[i-1].start, stack[i].start-1, arr+stack[i].start*size, &l);CHKERRQ(ierr);
        /* l == m-1 means sorted */
        if (l < m-1) {
          /* Search B for A[-1] insertion */
          ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[i].start, stack[i].start+stack[i].size-1, arr+(stack[i].start-1)*size, &r);CHKERRQ(ierr);
          if (m-l <= r-m) {
            ierr = PetscTimSortResizeBuffer_Private(buff, size, m-l+1);CHKERRQ(ierr);
            ierr = PetscTimSortMergeLo_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
          } else {
            ierr = PetscTimSortResizeBuffer_Private(buff, size, r-m+1);CHKERRQ(ierr);
            ierr = PetscTimSortMergeHi_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
          }
        }
        /* Update A with merge */
        stack[i-1].size += stack[i].size;
        --i;
      }
    } else {
      /* i > 2, i.e. C exists
       A = stack[i-2], B = stack[i-1], C = stack[i]; */
      if (stack[i-2].size <= stack[i-1].size+stack[i].size) {
        if (stack[i-2].size < stack[i].size) {
          /* merge B into A */
          m = stack[i-1].start;
          /* Search A for B[0] insertion */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[i-2].start, stack[i-1].start-1, arr+(stack[i-1].start)*size, &l);CHKERRQ(ierr);
          if (l < m-1) {
            /* Search B for A[-1] insertion */
            ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[i-1].start, stack[i-1].start+stack[i-1].size-1, arr+(stack[i-1].start-1)*size, &r);CHKERRQ(ierr);
            if (m-l <= r-m) {
              ierr = PetscTimSortResizeBuffer_Private(buff, size, m-l+1);CHKERRQ(ierr);
              ierr = PetscTimSortMergeLo_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
            } else {
              ierr = PetscTimSortResizeBuffer_Private(buff, size, r-m+1);CHKERRQ(ierr);
              ierr = PetscTimSortMergeHi_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
            }
          }
          /* Update A with merge */
          stack[i-2].size += stack[i-1].size;
          /* Push C up the stack */
          stack[i-1].start = stack[i].start;
          stack[i-1].size = stack[i].size;
        } else {
          /* merge C into B */
          mergeBC:
          m = stack[i].start;
          /* Search B for C[0] insertion */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[i-1].start, stack[i].start-1, arr+stack[i].start*size, &l);CHKERRQ(ierr);
          if (l < m-1) {
            /* Search C for B[-1] insertion */
            ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[i].start, stack[i].start+stack[i].size-1, arr+(stack[i].start-1)*size, &r);CHKERRQ(ierr);
            if (m-l <= r-m) {
              ierr = PetscTimSortResizeBuffer_Private(buff, size, m-l+1);CHKERRQ(ierr);
              ierr = PetscTimSortMergeLo_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
            } else {
              ierr = PetscTimSortResizeBuffer_Private(buff, size, r-m+1);CHKERRQ(ierr);
              ierr = PetscTimSortMergeHi_Private(arr, size, cmp, buff->ptr, l, m, r);CHKERRQ(ierr);
            }
          }
          /* Update B with merge */
          stack[i-1].size += stack[i].size;
        }
        --i;
      } else if (stack[i-1].size <= stack[i].size) {
        /* merge C into B */
        goto mergeBC;
      }
    }
    if (itemp == i) break;
  }
  *stacksize = i;
  PetscFunctionReturn(0);
}

/* March sequentially through the array building up a "run" of weakly increasing or strictly decreasing contiguous
 elements. Decreasing runs are reversed by swapping. If the run is less than minrun, artificially extend it via either
 binary insertion sort or regulat insertion sort */
static PetscErrorCode PetscTimSortBuildRun_Private(const PetscInt n, char *arr, const size_t size, PetscInt (*cmp)(const void *, const void *), PetscInt minrun, const PetscInt runstart, PetscInt *runend)
{
  PetscInt       re = PetscMin(runstart+minrun, n-1), ri = runstart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikely(runstart == n-1)) {*runend = runstart; PetscFunctionReturn(0);}
  /* guess whether run is ascending or descending and tally up the longest consecutive run. essentially a coinflip for random data */
  if ((*cmp)(arr+(ri+1)*size, arr+ri*size) < 0) {
    ++ri;
    while (ri < n-1) {
      if ((*cmp)(arr+(ri+1)*size, arr+ri*size) >= 0) break;
      ++ri;
    }
    {
      PetscInt lo = runstart, hi = ri;
      char     *t;
      ierr = PetscMalloc(size, &t);CHKERRQ(ierr);
      do{
        BYTESWAP(arr+lo*size,arr+hi*size,t,size);
        ++lo; --hi;
      } while (lo < hi);
      ierr = PetscFree(t);CHKERRQ(ierr);
    }
  } else {
    ++ri;
    while (ri < n-1) {
      if ((*cmp)(arr+(ri+1)*size, arr+ri*size) < 0) break;
      ++ri;
    }
  }
#if defined(PETSC_USE_DEBUG)
  ierr = PetscInfo1(NULL, "natural run length = %D\n", ri-runstart+1);CHKERRQ(ierr);
#endif
  if (ri < re) {
    /* the attempt failed, this section likely contains random data. If ri got close to minrun (within 50%) then we try
     binary search */
    if (ri-runstart <= minrun >> 1) {
      ++MIN_GALLOP_GLOBAL; /* didn't get close hedge our bets against random data */
      ierr = PetscInsertionSort_Private(arr, size, cmp, runstart, ri, re);CHKERRQ(ierr);
    } else {
      ierr = PetscBinaryInsertionSort_Private(arr, size, cmp, runstart, ri, re);CHKERRQ(ierr);
    }
    *runend = re;
  } else *runend = ri;
  PetscFunctionReturn(0);
}

/*
  PetscTimSort - Sorts an array in place in increasing order using Tim Peters adaptive sorting algorithm.

  Not Collective

  Input Parameters:
+ n    - number of values
. arr  - array to be sorted
. size - size in bytes of the datatype held in arr
- cmp  - function pointer to comparison function

  Output Parameters:
. arr  - sorted array

  Sample usage:
  The comparison function should take a left and right argument and return the signed difference between the two. The
 contents of the void pointers should be cast to the correct type inside the comparison function. For example when
 sorting an array of type "my_type" in increasing order.
.vb
  PetscInt my_increasing_comparison_function(const void *left, const void *right) {
    my_type l = *(my_type *) left, r = *(my_type *) right;
    return l < r ? -1 : l == r ? 0 : 1;
  }
.ve
  Then pass the function
.vb
  PetscTimSort(n, arr, sizeof(arr[0]), my_increasing_comparison_function)
.ve

  Notes:
  The comparison function must follow the qsort() comparison function paradigm, returning the signed difference between
  its arguments. If left < right : return -1, if left == right : return 0, if left > right : return 1. The user may also
 change or reverse the order of the sort by flipping the above. Note that stability of the sort is only guaranteed if
 the comparison function forms a valid trigraph.

  Timsort makes the assumption that input data is already likely partially ordered, or that it contains contiguous
  sections (termed 'runs') where the data is locally ordered (but not necessarily globally ordered). It therefore aims
 to select slices of the array in such a way that resulting mergesorts operate on near perfectly length-balanced
 arrays. To do so it repeatedly triggers attempts throughout to merge adjacent runs.

  Should one run continuously "win" a comparison the algorithm begins the "gallop" phase. It will aggressively
  search the "winner" for the location of the "losers" next entry (and vice versa) to copy all preceding elements into
  place in bulk. However if the data is truly unordered (as is the case with random data) the immense gains possible
  from these searches are expected __not__ to repay their costs. While adjacent arrays are almost all nearly the same
  size, they likely all contain similar data.

  A detailed description of the algorithm may be found here: https://bugs.python.org/file4451/timsort.txt

*/
PETSC_STATIC_INLINE PetscErrorCode PetscTimSort(const PetscInt n, void *arr, const size_t size, PetscInt (*cmp)(const void *, const void *))
{
  PetscInt           stacksize = 0, minrun, runstart = 0, runend = 0;
  PetscTimSortStack  runstack[128];
  PetscTimSortBuffer buff;
  char               *carr = (char *) arr;
  PetscErrorCode     ierr;
  /* stacksize  = log_phi(n) = log_2(n)/log_2(phi), so 128 is enough for ~5.614e26 elements.
   It is so unlikely that this limit is reached that this is __never__ checked for */

  PetscFunctionBegin;
  /* Compute minrun. Minrun should be (32, 65) such that N/minrun
   is a power of 2 or one plus a power of 2 */
  {
    PetscInt t = n, r = 0;
    /* r becomes 1 if the least significant bits contain at least one off bit */
    while (t >= 64) {
      r |= t & 1;
      t >>= 1;
    }
    minrun = t + r;
  }
  if (PetscUnlikelyDebug(minrun < 32 || minrun > 65)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Calculated minrun %D not in range (32,65)",minrun);
  ierr = PetscInfo1(NULL, "minrun = %D\n", minrun);CHKERRQ(ierr);
  ierr = PetscMalloc(minrun*size, &buff.ptr);CHKERRQ(ierr);
  buff.size = minrun*size;
  buff.maxsize = n*size;
  MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;
  while (runstart < n) {
    /* Check if additional entries are at least partially ordered and build natural run */
    ierr = PetscTimSortBuildRun_Private(n, carr, size, cmp, minrun, runstart, &runend);CHKERRQ(ierr);
    runstack[stacksize].start = runstart;
    runstack[stacksize].size = runend-runstart+1;
    ierr = PetscTimSortMergeCollapse_Private(carr, size, cmp, &buff, runstack, &stacksize);CHKERRQ(ierr);
    ++stacksize;
    runstart = runend+1;
  }
  /* Have been inside while, so discard last stacksize++ */
  --stacksize;
  ierr = PetscTimSortForceCollapse_Private(carr, size, cmp, &buff, runstack, stacksize);CHKERRQ(ierr);
  ierr = PetscFree(buff.ptr);CHKERRQ(ierr);
  MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscInt Compare_PetscMPIInt_Private(const void *left, const void *right)
{
  PetscMPIInt l = *(const PetscMPIInt *) left, r = *(const PetscMPIInt *) right;
  if (l < r) return -1;
  if (l == r) return 0;
  return 1;
}

PETSC_STATIC_INLINE PetscInt Compare_PetscInt_Private(const void *left, const void *right)
{
  PetscInt l = *(const PetscInt *) left, r = *(const PetscInt *) right;
  if (l < r) return -1;
  if (l == r) return 0;
  return 1;
}

/*
   PetscTryMethod - Queries an object for a method, if it exists then calls it.
              These are intended to be used only inside PETSc functions.

   Level: developer

.seealso: PetscUseMethod()
*/
#define  PetscTryMethod(obj,A,B,C) \
  0; do { PetscErrorCode (*_7_f)B, _7_ierr; \
    _7_ierr = PetscObjectQueryFunction((PetscObject)(obj),A,&_7_f);CHKERRQ(_7_ierr); \
    if (_7_f) {_7_ierr = (*_7_f)C;CHKERRQ(_7_ierr);} \
  } while(0)

/*
   PetscUseMethod - Queries an object for a method, if it exists then calls it, otherwise generates an error.
              These are intended to be used only inside PETSc functions.

   Level: developer

.seealso: PetscTryMethod()
*/
#define  PetscUseMethod(obj,A,B,C) \
  0; do { PetscErrorCode (*_7_f)B, _7_ierr; \
    _7_ierr = PetscObjectQueryFunction((PetscObject)(obj),A,&_7_f);CHKERRQ(_7_ierr); \
    if (_7_f) {_7_ierr = (*_7_f)C;CHKERRQ(_7_ierr);} \
    else SETERRQ1(PetscObjectComm((PetscObject)(obj)),PETSC_ERR_SUP,"Cannot locate function %s in object",A); \
  } while(0)

/*MC
   PetscObjectStateIncrease - Increases the state of any PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectStateIncrease(PetscObject obj)

   Logically Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example,
         PetscObjectStateIncrease((PetscObject)mat);

   Notes:
    object state is an integer which gets increased every time
   the object is changed internally. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   This routine is mostly for internal use by PETSc; a developer need only
   call it after explicit access to an object's internals. Routines such
   as VecSet() or MatScale() already call this routine. It is also called, as a
   precaution, in VecRestoreArray(), MatRestoreRow(), MatDenseRestoreArray().

   This routine is logically collective because state equality comparison needs to be possible without communication.

   Level: developer

   seealso: PetscObjectStateGet()

M*/
#define PetscObjectStateIncrease(obj) ((obj)->state++,0)

PETSC_EXTERN PetscErrorCode PetscObjectStateGet(PetscObject,PetscObjectState*);
PETSC_EXTERN PetscErrorCode PetscObjectStateSet(PetscObject,PetscObjectState);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataRegister(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseInt(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseIntstar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseReal(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseRealstar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseScalar(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectComposedDataIncreaseScalarstar(PetscObject);
PETSC_EXTERN PetscInt       PetscObjectComposedDataMax;
/*MC
   PetscObjectComposedDataSetInt - attach integer data to a PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetInt(PetscObject obj,int id,int data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be created through a call to  PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetInt(obj,id,data)                                      \
  ((((obj)->int_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseInt(obj)) ||  \
   ((obj)->intcomposeddata[id] = data,(obj)->intcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetInt - retrieve integer data attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetInt(PetscObject obj,int id,int data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters:
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetInt(obj,id,data,flag)                            \
  ((((obj)->intcomposedstate && ((obj)->intcomposedstate[id] == (obj)->state)) ?   \
   (data = (obj)->intcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetIntstar - attach integer array data to a PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetIntstar(PetscObject obj,int id,int *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetIntstar(obj,id,data)                                          \
  ((((obj)->intstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseIntstar(obj)) ||  \
   ((obj)->intstarcomposeddata[id] = data,(obj)->intstarcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetIntstar - retrieve integer array data
   attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetIntstar(PetscObject obj,int id,int *data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters:
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetIntstar(obj,id,data,flag)                               \
  ((((obj)->intstarcomposedstate && ((obj)->intstarcomposedstate[id] == (obj)->state)) ?  \
   (data = (obj)->intstarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetReal - attach real data to a PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj,int id,PetscReal data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetReal(obj,id,data)                                       \
  ((((obj)->real_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseReal(obj)) ||  \
   ((obj)->realcomposeddata[id] = data,(obj)->realcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetReal - retrieve real data attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj,int id,PetscReal data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters:
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetReal(obj,id,data,flag)                            \
  ((((obj)->realcomposedstate && ((obj)->realcomposedstate[id] == (obj)->state)) ?  \
   (data = (obj)->realcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetRealstar - attach real array data to a PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetRealstar(PetscObject obj,int id,PetscReal *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#define PetscObjectComposedDataSetRealstar(obj,id,data)                                           \
  ((((obj)->realstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseRealstar(obj)) ||  \
   ((obj)->realstarcomposeddata[id] = data, (obj)->realstarcomposedstate[id] = (obj)->state, 0))

/*MC
   PetscObjectComposedDataGetRealstar - retrieve real array data
   attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetRealstar(PetscObject obj,int id,PetscReal *data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters:
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#define PetscObjectComposedDataGetRealstar(obj,id,data,flag)                                \
  ((((obj)->realstarcomposedstate && ((obj)->realstarcomposedstate[id] == (obj)->state)) ?  \
   (data = (obj)->realstarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)

/*MC
   PetscObjectComposedDataSetScalar - attach scalar data to a PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetScalar(PetscObject obj,int id,PetscScalar data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataSetScalar(obj,id,data)                                        \
  ((((obj)->scalar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseScalar(obj)) || \
   ((obj)->scalarcomposeddata[id] = data,(obj)->scalarcomposedstate[id] = (obj)->state, 0))
#else
#define PetscObjectComposedDataSetScalar(obj,id,data) \
        PetscObjectComposedDataSetReal(obj,id,data)
#endif
/*MC
   PetscObjectComposedDataGetScalar - retrieve scalar data attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetScalar(PetscObject obj,int id,PetscScalar data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters:
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataGetScalar(obj,id,data,flag)                              \
  ((((obj)->scalarcomposedstate && ((obj)->scalarcomposedstate[id] == (obj)->state) ) ? \
   (data = (obj)->scalarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)
#else
#define PetscObjectComposedDataGetScalar(obj,id,data,flag)                             \
        PetscObjectComposedDataGetReal(obj,id,data,flag)
#endif

/*MC
   PetscObjectComposedDataSetScalarstar - attach scalar array data to a PetscObject

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataSetScalarstar(PetscObject obj,int id,PetscScalar *data)

   Not collective

   Input parameters:
+  obj - the object to which data is to be attached
.  id - the identifier for the data
-  data - the data to  be attached

   Notes
   The data identifier can best be determined through a call to
   PetscObjectComposedDataRegister()

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataSetScalarstar(obj,id,data)                                             \
  ((((obj)->scalarstar_idmax < PetscObjectComposedDataMax) && PetscObjectComposedDataIncreaseScalarstar(obj)) ||  \
   ((obj)->scalarstarcomposeddata[id] = data,(obj)->scalarstarcomposedstate[id] = (obj)->state, 0))
#else
#define PetscObjectComposedDataSetScalarstar(obj,id,data) \
        PetscObjectComposedDataSetRealstar(obj,id,data)
#endif
/*MC
   PetscObjectComposedDataGetScalarstar - retrieve scalar array data
   attached to an object

   Synopsis:
   #include "petsc/private/petscimpl.h"
   PetscErrorCode PetscObjectComposedDataGetScalarstar(PetscObject obj,int id,PetscScalar *data,PetscBool  flag)

   Not collective

   Input parameters:
+  obj - the object from which data is to be retrieved
-  id - the identifier for the data

   Output parameters:
+  data - the data to be retrieved
-  flag - PETSC_TRUE if the data item exists and is valid, PETSC_FALSE otherwise

   The 'data' and 'flag' variables are inlined, so they are not pointers.

   Level: developer
M*/
#if defined(PETSC_USE_COMPLEX)
#define PetscObjectComposedDataGetScalarstar(obj,id,data,flag)                                 \
  ((((obj)->scalarstarcomposedstate && ((obj)->scalarstarcomposedstate[id] == (obj)->state)) ? \
       (data = (obj)->scalarstarcomposeddata[id],flag = PETSC_TRUE) : (flag = PETSC_FALSE)),0)
#else
#define PetscObjectComposedDataGetScalarstar(obj,id,data,flag)         \
        PetscObjectComposedDataGetRealstar(obj,id,data,flag)
#endif

PETSC_EXTERN PetscMPIInt Petsc_Counter_keyval;
PETSC_EXTERN PetscMPIInt Petsc_InnerComm_keyval;
PETSC_EXTERN PetscMPIInt Petsc_OuterComm_keyval;
PETSC_EXTERN PetscMPIInt Petsc_Seq_keyval;
PETSC_EXTERN PetscMPIInt Petsc_ShmComm_keyval;

/*
  PETSc communicators have this attribute, see
  PetscCommDuplicate(), PetscCommDestroy(), PetscCommGetNewTag(), PetscObjectGetName()
*/
typedef struct {
  PetscMPIInt tag;              /* next free tag value */
  PetscInt    refcount;         /* number of references, communicator can be freed when this reaches 0 */
  PetscInt    namecount;        /* used to generate the next name, as in Vec_0, Mat_1, ... */
  PetscMPIInt *iflags;          /* length of comm size, shared by all calls to PetscCommBuildTwoSided_Allreduce/RedScatter on this comm */
} PetscCommCounter;

/*E
    PetscOffloadMask - indicates which memory (CPU, GPU, or none) contains valid data

   PETSC_OFFLOAD_UNALLOCATED  - no memory contains valid matrix entries; NEVER used for vectors
   PETSC_OFFLOAD_GPU - GPU has valid vector/matrix entries
   PETSC_OFFLOAD_CPU - CPU has valid vector/matrix entries
   PETSC_OFFLOAD_BOTH - Both GPU and CPU have valid vector/matrix entries and they match

   Level: developer
E*/
typedef enum {PETSC_OFFLOAD_UNALLOCATED=0x0,PETSC_OFFLOAD_CPU=0x1,PETSC_OFFLOAD_GPU=0x2,PETSC_OFFLOAD_BOTH=0x3} PetscOffloadMask;

typedef enum {STATE_BEGIN, STATE_PENDING, STATE_END} SRState;

typedef enum {PETSC_SR_REDUCE_SUM=0,PETSC_SR_REDUCE_MAX=1,PETSC_SR_REDUCE_MIN=2} PetscSRReductionType;

typedef struct {
  MPI_Comm    comm;
  MPI_Request request;
  PetscBool   async;
  PetscScalar *lvalues;     /* this are the reduced values before call to MPI_Allreduce() */
  PetscScalar *gvalues;     /* values after call to MPI_Allreduce() */
  void        **invecs;     /* for debugging only, vector/memory used with each op */
  PetscInt    *reducetype;  /* is particular value to be summed or maxed? */
  SRState     state;        /* are we calling xxxBegin() or xxxEnd()? */
  PetscInt    maxops;       /* total amount of space we have for requests */
  PetscInt    numopsbegin;  /* number of requests that have been queued in */
  PetscInt    numopsend;    /* number of requests that have been gotten by user */
} PetscSplitReduction;

PETSC_EXTERN PetscErrorCode PetscSplitReductionGet(MPI_Comm,PetscSplitReduction**);
PETSC_EXTERN PetscErrorCode PetscSplitReductionEnd(PetscSplitReduction*);
PETSC_EXTERN PetscErrorCode PetscSplitReductionExtend(PetscSplitReduction*);

#if !defined(PETSC_SKIP_SPINLOCK)
#if defined(PETSC_HAVE_THREADSAFETY)
#  if defined(PETSC_HAVE_CONCURRENCYKIT)
#if defined(__cplusplus)
/*  CK does not have extern "C" protection in their include files */
extern "C" {
#endif
#include <ck_spinlock.h>
#if defined(__cplusplus)
}
#endif
typedef ck_spinlock_t PetscSpinlock;
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockCreate(PetscSpinlock *ck_spinlock)
{
  ck_spinlock_init(ck_spinlock);
  return 0;
}
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockLock(PetscSpinlock *ck_spinlock)
{
  ck_spinlock_lock(ck_spinlock);
  return 0;
}
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockUnlock(PetscSpinlock *ck_spinlock)
{
  ck_spinlock_unlock(ck_spinlock);
  return 0;
}
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockDestroy(PetscSpinlock *ck_spinlock)
{
  return 0;
}
#  elif defined(PETSC_HAVE_OPENMP)

#include <omp.h>
typedef omp_lock_t PetscSpinlock;
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockCreate(PetscSpinlock *omp_lock)
{
  omp_init_lock(omp_lock);
  return 0;
}
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockLock(PetscSpinlock *omp_lock)
{
  omp_set_lock(omp_lock);
  return 0;
}
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockUnlock(PetscSpinlock *omp_lock)
{
  omp_unset_lock(omp_lock);
  return 0;
}
PETSC_STATIC_INLINE PetscErrorCode PetscSpinlockDestroy(PetscSpinlock *omp_lock)
{
  omp_destroy_lock(omp_lock);
  return 0;
}
#else
Thread safety requires either --with-openmp or --download-concurrencykit
#endif

#else
typedef int PetscSpinlock;
#define PetscSpinlockCreate(a)  0
#define PetscSpinlockLock(a)    0
#define PetscSpinlockUnlock(a)  0
#define PetscSpinlockDestroy(a) 0
#endif

#if defined(PETSC_HAVE_THREADSAFETY)
PETSC_INTERN PetscSpinlock PetscViewerASCIISpinLockOpen;
PETSC_INTERN PetscSpinlock PetscViewerASCIISpinLockStdout;
PETSC_INTERN PetscSpinlock PetscViewerASCIISpinLockStderr;
PETSC_INTERN PetscSpinlock PetscCommSpinLock;
#endif
#endif

PETSC_EXTERN PetscLogEvent PETSC_Barrier;
PETSC_EXTERN PetscLogEvent PETSC_BuildTwoSided;
PETSC_EXTERN PetscLogEvent PETSC_BuildTwoSidedF;
PETSC_EXTERN PetscBool     use_gpu_aware_mpi;

#if defined(PETSC_HAVE_ADIOS)
PETSC_EXTERN int64_t Petsc_adios_group;
#endif

#if defined(PETSC_HAVE_CUDA)
/* Has petsc initialized CUDA? One can use this flag to guard some CUDA calls, which may initialize CUDA runtime and incur a cost. */
PETSC_EXTERN PetscBool      PetscCUDAInitialized;
/* Initialize the CUDA device lazily just before creating the first CUDA object. */
PETSC_EXTERN PetscErrorCode PetscCUDAInitializeLazily(void);
#endif

#endif /* PETSCIMPL_H */
