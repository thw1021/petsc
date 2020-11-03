/*
     Include file for the matrix component of PETSc
*/
#ifndef PETSCMATPARTITIONING_H
#define PETSCMATPARTITIONING_H
#include <petscmat.h>

PETSC_EXTERN PetscFunctionList MatPartitioningList;

/*S
     MatPartitioning - Object for managing the partitioning of a matrix or graph

   Level: beginner

   Notes:
     There is also a PetscPartitioner object that provides the same functionality. It can utilize the MatPartitioning operations
     via PetscPartitionerSetType(p,PETSCPARTITIONERMATPARTITIONING)

   Developers Note:
     It is an extra maintainance and documentation cost to have two objects with the same functionality.

.seealso:  MatPartitioningCreate(), MatPartitioningType, MatColoring, MatGetOrdering()
S*/
typedef struct _p_MatPartitioning* MatPartitioning;

/*J
    MatPartitioningType - String with the name of a PETSc matrix partitioning

   Level: beginner
dm
.seealso: MatPartitioningCreate(), MatPartitioning
J*/
typedef const char* MatPartitioningType;
#define MATPARTITIONINGCURRENT   "current"
#define MATPARTITIONINGAVERAGE   "average"
#define MATPARTITIONINGSQUARE    "square"
#define MATPARTITIONINGPARMETIS  "parmetis"
#define MATPARTITIONINGCHACO     "chaco"
#define MATPARTITIONINGPARTY     "party"
#define MATPARTITIONINGPTSCOTCH  "ptscotch"
#define MATPARTITIONINGHIERARCH  "hierarch"

PETSC_EXTERN PetscErrorCode MatPartitioningCreate(MPI_Comm,MatPartitioning*);
PETSC_EXTERN PetscErrorCode MatPartitioningSetType(MatPartitioning,MatPartitioningType);
PETSC_EXTERN PetscErrorCode MatPartitioningSetNParts(MatPartitioning,PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningSetAdjacency(MatPartitioning,Mat);
PETSC_EXTERN PetscErrorCode MatPartitioningSetVertexWeights(MatPartitioning,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatPartitioningSetPartitionWeights(MatPartitioning,const PetscReal []);
PETSC_EXTERN PetscErrorCode MatPartitioningSetUseEdgeWeights(MatPartitioning,PetscBool);
PETSC_EXTERN PetscErrorCode MatPartitioningGetUseEdgeWeights(MatPartitioning,PetscBool*);
PETSC_EXTERN PetscErrorCode MatPartitioningApply(MatPartitioning,IS*);
PETSC_EXTERN PetscErrorCode MatPartitioningImprove(MatPartitioning,IS*);
PETSC_EXTERN PetscErrorCode MatPartitioningViewImbalance(MatPartitioning,IS);
PETSC_EXTERN PetscErrorCode MatPartitioningApplyND(MatPartitioning,IS*);
PETSC_EXTERN PetscErrorCode MatPartitioningDestroy(MatPartitioning*);
PETSC_EXTERN PetscErrorCode MatPartitioningRegister(const char[],PetscErrorCode (*)(MatPartitioning));
PETSC_EXTERN PetscErrorCode MatPartitioningView(MatPartitioning,PetscViewer);
PETSC_EXTERN PetscErrorCode MatPartitioningViewFromOptions(MatPartitioning,PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode MatPartitioningSetFromOptions(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningGetType(MatPartitioning,MatPartitioningType*);

PETSC_EXTERN PetscErrorCode MatPartitioningParmetisSetRepartition(MatPartitioning part);
PETSC_EXTERN PetscErrorCode MatPartitioningParmetisGetEdgeCut(MatPartitioning, PetscInt *);

typedef enum { MP_CHACO_MULTILEVEL=1,MP_CHACO_SPECTRAL=2,MP_CHACO_LINEAR=4,MP_CHACO_RANDOM=5,MP_CHACO_SCATTERED=6 } MPChacoGlobalType;
PETSC_EXTERN const char *const MPChacoGlobalTypes[];
typedef enum { MP_CHACO_KERNIGHAN=1,MP_CHACO_NONE=2 } MPChacoLocalType;
PETSC_EXTERN const char *const MPChacoLocalTypes[];
typedef enum { MP_CHACO_LANCZOS=0,MP_CHACO_RQI=1 } MPChacoEigenType;
PETSC_EXTERN const char *const MPChacoEigenTypes[];

PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetGlobal(MatPartitioning,MPChacoGlobalType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetGlobal(MatPartitioning,MPChacoGlobalType*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetLocal(MatPartitioning,MPChacoLocalType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetLocal(MatPartitioning,MPChacoLocalType*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetCoarseLevel(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenSolver(MatPartitioning,MPChacoEigenType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenSolver(MatPartitioning,MPChacoEigenType*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenTol(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenTol(MatPartitioning,PetscReal*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenNumber(MatPartitioning,PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenNumber(MatPartitioning,PetscInt*);

#define MP_PARTY_OPT "opt"
#define MP_PARTY_LIN "lin"
#define MP_PARTY_SCA "sca"
#define MP_PARTY_RAN "ran"
#define MP_PARTY_GBF "gbf"
#define MP_PARTY_GCF "gcf"
#define MP_PARTY_BUB "bub"
#define MP_PARTY_DEF "def"
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetGlobal(MatPartitioning,const char*);
#define MP_PARTY_HELPFUL_SETS "hs"
#define MP_PARTY_KERNIGHAN_LIN "kl"
#define MP_PARTY_NONE "no"
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetLocal(MatPartitioning,const char*);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetCoarseLevel(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetBipart(MatPartitioning,PetscBool);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetMatchOptimization(MatPartitioning,PetscBool);

typedef enum { MP_PTSCOTCH_DEFAULT,MP_PTSCOTCH_QUALITY,MP_PTSCOTCH_SPEED,MP_PTSCOTCH_BALANCE,MP_PTSCOTCH_SAFETY,MP_PTSCOTCH_SCALABILITY } MPPTScotchStrategyType;
PETSC_EXTERN const char *const MPPTScotchStrategyTypes[];

PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchSetImbalance(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchGetImbalance(MatPartitioning,PetscReal*);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchSetStrategy(MatPartitioning,MPPTScotchStrategyType);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchGetStrategy(MatPartitioning,MPPTScotchStrategyType*);

/*
 * hierarchical partitioning
 */
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalGetFineparts(MatPartitioning,IS*);
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalGetCoarseparts(MatPartitioning,IS*);
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalSetNcoarseparts(MatPartitioning,PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalSetNfineparts(MatPartitioning, PetscInt);

PETSC_EXTERN PetscErrorCode MatMeshToCellGraph(Mat,PetscInt,Mat*);
#endif
