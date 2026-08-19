/*
     Include file for the matrix component of PETSc
*/
#pragma once

#include <petscvec.h>
#include <petscmattypes.h>

/* SUBMANSEC = Mat */

PETSC_EXTERN PetscErrorCode MatGetFactor(Mat, MatSolverType, MatFactorType, Mat *);
PETSC_EXTERN PetscErrorCode MatGetFactorAvailable(Mat, MatSolverType, MatFactorType, PetscBool *);
PETSC_EXTERN PetscErrorCode MatFactorGetCanUseOrdering(Mat, PetscBool *);
PETSC_DEPRECATED_FUNCTION(3, 15, 0, "MatFactorGetCanUseOrdering()", ) static inline PetscErrorCode MatFactorGetUseOrdering(Mat A, PetscBool *b)
{
  return MatFactorGetCanUseOrdering(A, b);
}
PETSC_EXTERN PetscErrorCode MatFactorGetSolverType(Mat, MatSolverType *);
PETSC_EXTERN PetscErrorCode MatGetFactorType(Mat, MatFactorType *);
PETSC_EXTERN PetscErrorCode MatSetFactorType(Mat, MatFactorType);

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister(MatSolverType, MatType, MatFactorType, MatSolverFn *);
PETSC_EXTERN PetscErrorCode MatSolverTypeGet(MatSolverType, MatType, MatFactorType, PetscBool *, PetscBool *, MatSolverFn **);
typedef MatSolverType       MatSolverPackage PETSC_DEPRECATED_TYPEDEF(3, 9, 0, "MatSolverType", );
PETSC_DEPRECATED_FUNCTION(3, 9, 0, "MatSolverTypeRegister()", ) static inline PetscErrorCode MatSolverPackageRegister(MatSolverType stype, MatType mtype, MatFactorType ftype, MatSolverFn *f)
{
  return MatSolverTypeRegister(stype, mtype, ftype, f);
}
PETSC_DEPRECATED_FUNCTION(3, 9, 0, "MatSolverTypeGet()", ) static inline PetscErrorCode MatSolverPackageGet(MatSolverType stype, MatType mtype, MatFactorType ftype, PetscBool *foundmtype, PetscBool *foundstype, MatSolverFn **f)
{
  return MatSolverTypeGet(stype, mtype, ftype, foundmtype, foundstype, f);
}

PETSC_EXTERN PetscErrorCode MatProductCreate(Mat, Mat, Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatProductCreateWithMat(Mat, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatProductSetType(Mat, MatProductType);
PETSC_EXTERN PetscErrorCode MatProductSetAlgorithm(Mat, MatProductAlgorithm);
PETSC_EXTERN PetscErrorCode MatProductGetAlgorithm(Mat, MatProductAlgorithm *);
PETSC_EXTERN PetscErrorCode MatProductSetFill(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatProductSetFromOptions(Mat);
PETSC_EXTERN PetscErrorCode MatProductSymbolic(Mat);
PETSC_EXTERN PetscErrorCode MatProductNumeric(Mat);
PETSC_EXTERN PetscErrorCode MatProductReplaceMats(Mat, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatProductClear(Mat);
PETSC_EXTERN PetscErrorCode MatProductView(Mat, PetscViewer);
PETSC_EXTERN PetscErrorCode MatProductGetType(Mat, MatProductType *);
PETSC_EXTERN PetscErrorCode MatProductGetMats(Mat, Mat *, Mat *, Mat *);

/* Logging support */
#define MAT_FILE_CLASSID 1211216 /* used to indicate matrices in binary files */
PETSC_EXTERN PetscClassId MAT_CLASSID;
PETSC_EXTERN PetscClassId MAT_COLORING_CLASSID;
PETSC_EXTERN PetscClassId MAT_FDCOLORING_CLASSID;
PETSC_EXTERN PetscClassId MAT_TRANSPOSECOLORING_CLASSID;
PETSC_EXTERN PetscClassId MAT_PARTITIONING_CLASSID;
PETSC_EXTERN PetscClassId MAT_COARSEN_CLASSID;
PETSC_EXTERN PetscClassId MAT_NULLSPACE_CLASSID;
PETSC_EXTERN PetscClassId MATMFFD_CLASSID;

PETSC_EXTERN PetscErrorCode MatInitializePackage(void);
PETSC_EXTERN PetscErrorCode MatFinalizePackage(void);

PETSC_EXTERN PetscErrorCode MatCreate(MPI_Comm, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateFromOptions(MPI_Comm, const char *, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatSetSizes(Mat, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode MatSetType(Mat, MatType);
PETSC_EXTERN PetscErrorCode MatGetVecType(Mat, VecType *);
PETSC_EXTERN PetscErrorCode MatSetVecType(Mat, VecType);
PETSC_EXTERN PetscErrorCode MatSetFromOptions(Mat);
PETSC_EXTERN PetscErrorCode MatViewFromOptions(Mat, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode MatRegister(const char[], PetscErrorCode (*)(Mat));
PETSC_EXTERN PetscErrorCode MatRegisterRootName(const char[], const char[], const char[]);
PETSC_EXTERN PetscErrorCode MatSetOptionsPrefix(Mat, const char[]);
PETSC_EXTERN PetscErrorCode MatSetOptionsPrefixFactor(Mat, const char[]);
PETSC_EXTERN PetscErrorCode MatAppendOptionsPrefixFactor(Mat, const char[]);
PETSC_EXTERN PetscErrorCode MatAppendOptionsPrefix(Mat, const char[]);
PETSC_EXTERN PetscErrorCode MatGetOptionsPrefix(Mat, const char *[]);
PETSC_EXTERN PetscErrorCode MatGetState(Mat, PetscObjectState *);
PETSC_EXTERN PetscErrorCode MatSetErrorIfFailure(Mat, PetscBool);

PETSC_EXTERN PetscFunctionList MatList;
PETSC_EXTERN PetscFunctionList MatColoringList;
PETSC_EXTERN PetscFunctionList MatPartitioningList;
PETSC_EXTERN PetscFunctionList MatMeshToCellGraphList;

#if PetscDefined(HAVE_MKL_SPARSE)
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJMKL(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJMKL(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateBAIJMKL(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqBAIJMKL(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
#endif

PETSC_EXTERN PetscErrorCode MatCreateMPIAIJPERM(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJPERM(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);

PETSC_EXTERN PetscErrorCode MatCreateSeqSELL(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSELL(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatSeqSELLSetPreallocation(Mat, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPISELLSetPreallocation(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqSELLGetFillRatio(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatSeqSELLGetMaxSliceWidth(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatSeqSELLGetAvgSliceWidth(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatSeqSELLSetSliceHeight(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSeqSELLGetVarSliceSize(Mat, PetscReal *);

PETSC_EXTERN PetscErrorCode MatCreateSeqAIJSELL(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSELL(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatMPISELLGetLocalMatCondensed(Mat, MatReuse, IS *, IS *, Mat *);
PETSC_EXTERN PetscErrorCode MatMPISELLGetSeqSELL(Mat, Mat *, Mat *, const PetscInt *[]);

PETSC_EXTERN PetscErrorCode MatCreateSeqDense(MPI_Comm, PetscInt, PetscInt, PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateDense(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJ(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateAIJ(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatUpdateMPIAIJWithArrays(Mat, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatUpdateMPIAIJWithArray(Mat, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJWithSplitArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscScalar[], PetscInt[], PetscInt[], PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJWithSeqAIJ(MPI_Comm, PetscInt, PetscInt, Mat, Mat, PetscInt *, Mat *);

PETSC_EXTERN PetscErrorCode MatCreateSeqBAIJ(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateBAIJ(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIBAIJWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], Mat *);

PETSC_EXTERN PetscErrorCode MatSetPreallocationCOO(Mat, PetscCount, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSetPreallocationCOOLocal(Mat, PetscCount, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSetValuesCOO(Mat, const PetscScalar[], InsertMode);

PETSC_EXTERN PetscErrorCode MatCreateMPIAdj(MPI_Comm, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqSBAIJ(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);

PETSC_EXTERN PetscErrorCode MatCreateSBAIJ(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPISBAIJWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatSeqSBAIJSetPreallocationCSR(Mat, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPISBAIJSetPreallocationCSR(Mat, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatXAIJSetPreallocation(Mat, PetscInt, const PetscInt[], const PetscInt[], const PetscInt[], const PetscInt[]);

PETSC_EXTERN PetscErrorCode MatCreateShell(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscCtx, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateCentering(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateNormal(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateNormalHermitian(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLRC(Mat, Mat, Vec, Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatLRCGetMats(Mat, Mat *, Mat *, Vec *, Mat *);
PETSC_EXTERN PetscErrorCode MatLRCSetMats(Mat, Mat, Mat, Vec, Mat);
PETSC_EXTERN PetscErrorCode MatCreateIS(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, ISLocalToGlobalMapping, ISLocalToGlobalMapping, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJCRL(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJCRL(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);

PETSC_EXTERN PetscErrorCode MatCreateScatter(MPI_Comm, VecScatter, Mat *);
PETSC_EXTERN PetscErrorCode MatScatterSetVecScatter(Mat, VecScatter);
PETSC_EXTERN PetscErrorCode MatScatterGetVecScatter(Mat, VecScatter *);
PETSC_EXTERN PetscErrorCode MatCreateBlockMat(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt *, Mat *);
PETSC_EXTERN PetscErrorCode MatCompositeAddMat(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatCompositeMerge(Mat);
PETSC_EXTERN PetscErrorCode MatCompositeSetMergeType(Mat, MatCompositeMergeType);
PETSC_EXTERN PetscErrorCode MatCreateComposite(MPI_Comm, PetscInt, const Mat *, Mat *);
PETSC_EXTERN PetscErrorCode MatCompositeSetType(Mat, MatCompositeType);
PETSC_EXTERN PetscErrorCode MatCompositeGetType(Mat, MatCompositeType *);
PETSC_EXTERN PetscErrorCode MatCompositeSetMatStructure(Mat, MatStructure);
PETSC_EXTERN PetscErrorCode MatCompositeGetMatStructure(Mat, MatStructure *);
PETSC_EXTERN PetscErrorCode MatCompositeGetNumberMat(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatCompositeGetMat(Mat, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCompositeSetScalings(Mat, const PetscScalar *);

PETSC_EXTERN PetscErrorCode MatCreateFFT(MPI_Comm, PetscInt, const PetscInt[], MatType, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqCUFFT(MPI_Comm, PetscInt, const PetscInt[], Mat *);

PETSC_EXTERN PetscErrorCode MatCreateTranspose(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatTransposeGetMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateHermitianTranspose(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatHermitianTransposeGetMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatNormalGetMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatNormalHermitianGetMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSubMatrixVirtual(Mat, IS, IS, Mat *);
PETSC_EXTERN PetscErrorCode MatSubMatrixVirtualUpdate(Mat, Mat, IS, IS);
PETSC_EXTERN PetscErrorCode MatCreateLocalRef(Mat, IS, IS, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateConstantDiagonal(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateDiagonal(Vec, Mat *);
PETSC_EXTERN PetscErrorCode MatDiagonalSetDiagonal(Mat, Vec);
PETSC_EXTERN PetscErrorCode MatDiagonalGetDiagonal(Mat, Vec *);
PETSC_EXTERN PetscErrorCode MatDiagonalRestoreDiagonal(Mat, Vec *);
PETSC_EXTERN PetscErrorCode MatDiagonalGetInverseDiagonal(Mat, Vec *);
PETSC_EXTERN PetscErrorCode MatDiagonalRestoreInverseDiagonal(Mat, Vec *);
PETSC_EXTERN PetscErrorCode MatConstantDiagonalGetConstant(Mat, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatGetCurrentMemType(Mat, PetscMemType *);

#if PetscDefined(HAVE_HYPRE)
PETSC_EXTERN PetscErrorCode MatHYPRESetPreallocation(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
#endif

PETSC_EXTERN PetscErrorCode MatPythonSetType(Mat, const char[]);
PETSC_EXTERN PetscErrorCode MatPythonGetType(Mat, const char *[]);
PETSC_EXTERN PetscErrorCode MatPythonCreate(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const char[], Mat *);

PETSC_EXTERN PetscErrorCode MatResetPreallocation(Mat);
PETSC_EXTERN PetscErrorCode MatResetHash(Mat);
PETSC_EXTERN PetscErrorCode MatSetUp(Mat);
PETSC_EXTERN PetscErrorCode MatDestroy(Mat *);
PETSC_EXTERN PetscErrorCode MatGetNonzeroState(Mat, PetscObjectState *);

PETSC_EXTERN PetscErrorCode MatConjugate(Mat);
PETSC_EXTERN PetscErrorCode MatRealPart(Mat);
PETSC_EXTERN PetscErrorCode MatImaginaryPart(Mat);
PETSC_EXTERN PetscErrorCode MatGetDiagonalBlock(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatGetMultPetscSF(Mat, PetscSF *);
PETSC_EXTERN PetscErrorCode MatGetTrace(Mat, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatInvertBlockDiagonal(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatInvertVariableBlockDiagonal(Mat, PetscInt, const PetscInt *, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatInvertBlockDiagonalMat(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatInvertVariableBlockEnvelope(Mat, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatComputeVariableBlockEnvelope(Mat);

PETSC_EXTERN PetscErrorCode MatSetValues(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesIS(Mat, IS, IS, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesBlocked(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesRow(Mat, PetscInt, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSetValuesRowLocal(Mat, PetscInt, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSetValuesBatch(Mat, PetscInt, PetscInt, PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSetRandom(Mat, PetscRandom);

PETSC_EXTERN PetscErrorCode MatSetValuesStencil(Mat, PetscInt, const MatStencil[], PetscInt, const MatStencil[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesBlockedStencil(Mat, PetscInt, const MatStencil[], PetscInt, const MatStencil[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode MatSetStencil(Mat, PetscInt, const PetscInt[], const PetscInt[], PetscInt);

PETSC_EXTERN PetscErrorCode MatAssemblyBegin(Mat, MatAssemblyType);
PETSC_EXTERN PetscErrorCode MatAssemblyEnd(Mat, MatAssemblyType);
PETSC_EXTERN PetscErrorCode MatAssembled(Mat, PetscBool *);

PETSC_EXTERN PetscErrorCode MatSetOption(Mat, MatOption, PetscBool);
PETSC_EXTERN PetscErrorCode MatGetOption(Mat, MatOption, PetscBool *);
PETSC_EXTERN PetscErrorCode MatPropagateSymmetryOptions(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatGetType(Mat, MatType *);

PETSC_EXTERN PetscErrorCode    MatGetValues(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode    MatGetRow(Mat, PetscInt, PetscInt *, const PetscInt *[], const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatRestoreRow(Mat, PetscInt, PetscInt *, const PetscInt *[], const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatGetRowUpperTriangular(Mat);
PETSC_EXTERN PetscErrorCode    MatRestoreRowUpperTriangular(Mat);
PETSC_EXTERN PetscErrorCode    MatGetColumnVector(Mat, Vec, PetscInt);
PETSC_EXTERN PetscErrorCode    MatSeqAIJGetArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqAIJGetArrayRead(Mat, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqAIJGetArrayWrite(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqAIJRestoreArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqAIJRestoreArrayRead(Mat, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqAIJRestoreArrayWrite(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqAIJGetMaxRowNonzeros(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode    MatSeqAIJSetValuesLocalFast(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode    MatSeqAIJSetType(Mat, MatType);
PETSC_EXTERN PetscErrorCode    MatSeqAIJKron(Mat, Mat, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode    MatSeqAIJRegister(const char[], PetscErrorCode (*)(Mat, MatType, MatReuse, Mat *));
PETSC_EXTERN PetscFunctionList MatSeqAIJList;
PETSC_EXTERN PetscErrorCode    MatSeqBAIJGetArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqBAIJRestoreArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqSBAIJGetArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatSeqSBAIJRestoreArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseGetArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseRestoreArray(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDensePlaceArray(Mat, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode    MatDenseReplaceArray(Mat, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode    MatDenseReplaceArrayWithMemType(Mat, PetscMemType, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode    MatDenseResetArray(Mat);
PETSC_EXTERN PetscErrorCode    MatDenseGetArrayRead(Mat, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseRestoreArrayRead(Mat, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseGetArrayWrite(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseRestoreArrayWrite(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseGetArrayAndMemType(Mat, PetscScalar *[], PetscMemType *);
PETSC_EXTERN PetscErrorCode    MatDenseRestoreArrayAndMemType(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseGetArrayReadAndMemType(Mat, const PetscScalar *[], PetscMemType *);
PETSC_EXTERN PetscErrorCode    MatDenseRestoreArrayReadAndMemType(Mat, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatDenseGetArrayWriteAndMemType(Mat, PetscScalar *[], PetscMemType *);
PETSC_EXTERN PetscErrorCode    MatDenseRestoreArrayWriteAndMemType(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode    MatGetBlockSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode    MatSetBlockSize(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode    MatGetBlockSizes(Mat, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode    MatSetBlockSizes(Mat, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode    MatSetBlockSizesFromMats(Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode    MatSetVariableBlockSizes(Mat, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode    MatGetVariableBlockSizes(Mat, PetscInt *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode    MatSelectVariableBlockSizes(Mat, Mat, IS);

PETSC_EXTERN PetscErrorCode MatDenseGetColumn(Mat, PetscInt, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode MatDenseRestoreColumn(Mat, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode MatDenseGetColumnVec(Mat, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode MatDenseRestoreColumnVec(Mat, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode MatDenseGetColumnVecRead(Mat, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode MatDenseRestoreColumnVecRead(Mat, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode MatDenseGetColumnVecWrite(Mat, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode MatDenseRestoreColumnVecWrite(Mat, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode MatDenseGetSubMatrix(Mat, PetscInt, PetscInt, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatDenseRestoreSubMatrix(Mat, Mat *);

PETSC_EXTERN PetscErrorCode MatMult(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMultDiagonalBlock(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMultAdd(Mat, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMultTranspose(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMultHermitianTranspose(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatADot(Mat, Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatANorm(Mat, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode MatIsTranspose(Mat, Mat, PetscReal, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsHermitianTranspose(Mat, Mat, PetscReal, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultTransposeAdd(Mat, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMultHermitianTransposeAdd(Mat, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMatSolve(Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatMatSolveTranspose(Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatMatTransposeSolve(Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatResidual(Mat, Vec, Vec, Vec);

PETSC_EXTERN PetscErrorCode MatConvert(Mat, MatType, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatDuplicate(Mat, MatDuplicateOption, Mat *);

PETSC_EXTERN PetscErrorCode MatCopy(Mat, Mat, MatStructure);
PETSC_EXTERN PetscErrorCode MatView(Mat, PetscViewer);
PETSC_EXTERN PetscErrorCode MatIsSymmetric(Mat, PetscReal, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsStructurallySymmetric(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsHermitian(Mat, PetscReal, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsSymmetricKnown(Mat, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsHermitianKnown(Mat, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsStructurallySymmetricKnown(Mat, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsSPDKnown(Mat, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatLoad(Mat, PetscViewer);

PETSC_EXTERN PetscErrorCode MatGetRowIJ(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
PETSC_EXTERN PetscErrorCode MatRestoreRowIJ(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
PETSC_EXTERN PetscErrorCode MatGetColumnIJ(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
PETSC_EXTERN PetscErrorCode MatRestoreColumnIJ(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);

PETSC_EXTERN PetscErrorCode MatGetInfo(Mat, MatInfoType, MatInfo *);
PETSC_EXTERN PetscErrorCode MatGetDiagonal(Mat, Vec);
PETSC_EXTERN PetscErrorCode MatGetRowMax(Mat, Vec, PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowMin(Mat, Vec, PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowMaxAbs(Mat, Vec, PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowSumAbs(Mat, Vec);
PETSC_EXTERN PetscErrorCode MatGetRowMinAbs(Mat, Vec, PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowSum(Mat, Vec);
PETSC_EXTERN PetscErrorCode MatTranspose(Mat, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatTransposeSymbolic(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatTransposeSetPrecursor(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatHermitianTranspose(Mat, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatPermute(Mat, IS, IS, Mat *);
PETSC_EXTERN PetscErrorCode MatDiagonalScale(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatDiagonalSet(Mat, Vec, InsertMode);

PETSC_EXTERN PetscErrorCode MatEqual(Mat, Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultEqual(Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultAddEqual(Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultTransposeEqual(Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultTransposeAddEqual(Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultHermitianTransposeEqual(Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultHermitianTransposeAddEqual(Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMatMultEqual(Mat, Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatTransposeMatMultEqual(Mat, Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMatTransposeMultEqual(Mat, Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatPtAPMultEqual(Mat, Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatRARtMultEqual(Mat, Mat, Mat, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsLinear(Mat, PetscInt, PetscBool *);

PETSC_EXTERN PetscErrorCode MatNorm(Mat, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode MatNormApproximate(Mat, NormType, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnNorms(Mat, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnSums(Mat, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatGetColumnSumsRealPart(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnSumsImaginaryPart(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnMeans(Mat, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatGetColumnMeansRealPart(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnMeansImaginaryPart(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnReductions(Mat, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode MatZeroEntries(Mat);
PETSC_EXTERN PetscErrorCode MatSetInf(Mat);
PETSC_EXTERN PetscErrorCode MatZeroRows(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsIS(Mat, IS, PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsStencil(Mat, PetscInt, const MatStencil[], PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsStencil(Mat, PetscInt, const MatStencil[], PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumns(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsIS(Mat, IS, PetscScalar, Vec, Vec);

PETSC_EXTERN PetscErrorCode MatGetSize(Mat, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatGetLocalSize(Mat, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRange(Mat, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRanges(Mat, const PetscInt **);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRangeColumn(Mat, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRangesColumn(Mat, const PetscInt **);
PETSC_EXTERN PetscErrorCode MatGetOwnershipIS(Mat, IS *, IS *);

PETSC_EXTERN PetscErrorCode MatCreateSubMatrices(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_DEPRECATED_FUNCTION(3, 8, 0, "MatCreateSubMatrices()", ) static inline PetscErrorCode MatGetSubMatrices(Mat mat, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  return MatCreateSubMatrices(mat, n, irow, icol, scall, submat);
}
PETSC_EXTERN PetscErrorCode MatCreateSubMatricesMPI(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_DEPRECATED_FUNCTION(3, 8, 0, "MatCreateSubMatricesMPI()", ) static inline PetscErrorCode MatGetSubMatricesMPI(Mat mat, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  return MatCreateSubMatricesMPI(mat, n, irow, icol, scall, submat);
}
PETSC_EXTERN PetscErrorCode MatDestroyMatrices(PetscInt, Mat *[]);
PETSC_EXTERN PetscErrorCode MatDestroySubMatrices(PetscInt, Mat *[]);
PETSC_EXTERN PetscErrorCode MatCreateSubMatrix(Mat, IS, IS, MatReuse, Mat *);
PETSC_DEPRECATED_FUNCTION(3, 8, 0, "MatCreateSubMatrix()", ) static inline PetscErrorCode MatGetSubMatrix(Mat mat, IS isrow, IS iscol, MatReuse cll, Mat *newmat)
{
  return MatCreateSubMatrix(mat, isrow, iscol, cll, newmat);
}
PETSC_EXTERN PetscErrorCode MatGetLocalSubMatrix(Mat, IS, IS, Mat *);
PETSC_EXTERN PetscErrorCode MatRestoreLocalSubMatrix(Mat, IS, IS, Mat *);
PETSC_EXTERN PetscErrorCode MatGetSeqNonzeroStructure(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatDestroySeqNonzeroStructure(Mat *);

PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSumSeqAIJ(MPI_Comm, Mat, PetscInt, PetscInt, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSumSeqAIJSymbolic(MPI_Comm, Mat, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSumSeqAIJNumeric(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetLocalMat(Mat, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatAIJGetLocalMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetLocalMatCondensed(Mat, MatReuse, IS *, IS *, Mat *);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetLocalMatMerge(Mat, MatReuse, IS *, Mat *);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetNumberNonzeros(Mat, PetscCount *);
PETSC_EXTERN PetscErrorCode MatGetBrowsOfAcols(Mat, Mat, MatReuse, IS *, IS *, Mat *);
PETSC_EXTERN PetscErrorCode MatGetGhosts(Mat, PetscInt *, const PetscInt *[]);

PETSC_EXTERN PetscErrorCode MatIncreaseOverlap(Mat, PetscInt, IS[], PetscInt);
PETSC_EXTERN PetscErrorCode MatIncreaseOverlapSplit(Mat, PetscInt, IS[], PetscInt);
PETSC_EXTERN PetscErrorCode MatMPIAIJSetUseScalableIncreaseOverlap(Mat, PetscBool);

PETSC_EXTERN PetscErrorCode MatMatMult(Mat, Mat, MatReuse, PetscReal, Mat *);

PETSC_EXTERN PetscErrorCode MatMatMatMult(Mat, Mat, Mat, MatReuse, PetscReal, Mat *);
PETSC_EXTERN PetscErrorCode MatGalerkin(Mat, Mat, Mat, MatReuse, PetscReal, Mat *);

PETSC_EXTERN PetscErrorCode MatPtAP(Mat, Mat, MatReuse, PetscReal, Mat *);
PETSC_EXTERN PetscErrorCode MatRARt(Mat, Mat, MatReuse, PetscReal, Mat *);

PETSC_EXTERN PetscErrorCode MatTransposeMatMult(Mat, Mat, MatReuse, PetscReal, Mat *);
PETSC_EXTERN PetscErrorCode MatMatTransposeMult(Mat, Mat, MatReuse, PetscReal, Mat *);

PETSC_EXTERN PetscErrorCode MatAXPY(Mat, PetscScalar, Mat, MatStructure);
PETSC_EXTERN PetscErrorCode MatAYPX(Mat, PetscScalar, Mat, MatStructure);

PETSC_EXTERN PetscErrorCode MatScale(Mat, PetscScalar);
PETSC_EXTERN PetscErrorCode MatShift(Mat, PetscScalar);

PETSC_EXTERN PetscErrorCode MatSetLocalToGlobalMapping(Mat, ISLocalToGlobalMapping, ISLocalToGlobalMapping);
PETSC_EXTERN PetscErrorCode MatGetLocalToGlobalMapping(Mat, ISLocalToGlobalMapping *, ISLocalToGlobalMapping *);
PETSC_EXTERN PetscErrorCode MatGetLayouts(Mat, PetscLayout *, PetscLayout *);
PETSC_EXTERN PetscErrorCode MatSetLayouts(Mat, PetscLayout, PetscLayout);
PETSC_EXTERN PetscErrorCode MatZeroRowsLocal(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsLocalIS(Mat, IS, PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsLocal(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsLocalIS(Mat, IS, PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatGetValuesLocal(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSetValuesLocal(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesBlockedLocal(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);

PETSC_EXTERN PetscErrorCode MatStashSetInitialSize(Mat, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode MatStashGetInfo(Mat, PetscInt *, PetscInt *, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode MatInterpolate(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatInterpolateAdd(Mat, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatRestrict(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMatInterpolate(Mat, Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatMatInterpolateAdd(Mat, Mat, Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatMatRestrict(Mat, Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateVecs(Mat, Vec *, Vec *);
PETSC_DEPRECATED_FUNCTION(3, 6, 0, "MatCreateVecs()", ) static inline PetscErrorCode MatGetVecs(Mat mat, Vec *x, Vec *y)
{
  return MatCreateVecs(mat, x, y);
}
PETSC_EXTERN PetscErrorCode MatCreateRedundantMatrix(Mat, PetscInt, MPI_Comm, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatGetMultiProcBlock(Mat, MPI_Comm, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatFindZeroDiagonals(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatFindOffBlockDiagonalEntries(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat(MPI_Comm, Mat, PetscInt, MatReuse, Mat *);

PETSC_EXTERN PetscErrorCode MatCopyHashToXAIJ(Mat A, Mat B);
/*@
   MatSetValue - Set a single entry into a matrix.

   Not Collective

   Input Parameters:
+  mat  - the matrix
.  i    - the row location of the entry
.  j    - the column location of the entry
.  va   - the value to insert
-  mode - either `INSERT_VALUES` or `ADD_VALUES`

   Level: beginner

   Notes:
   This value may be cached, so `MatAssemblyBegin()` and `MatAssemblyEnd()`
   MUST be called after all calls to `MatSetValue()` have been completed.

   For efficiency one should use `MatSetValues()` and set several values simultaneously.

.seealso: [](ch_matrices), `Mat`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `InsertMode`, `MatGetValue()`, `MatSetValues()`,
          `MatSetValueLocal()`, `MatSetValuesLocal()`
@*/
static inline PetscErrorCode MatSetValue(Mat mat, PetscInt i, PetscInt j, PetscScalar va, InsertMode mode)
{
  return MatSetValues(mat, 1, &i, 1, &j, &va, mode);
}

/*@
   MatGetValue - Gets a single value from a matrix

   Not Collective; can only return a value owned by the given process

   Input Parameters:
+  mat - the matrix
.  row - the row location of the entry
-  col - the column location of the entry

   Output Parameter:
.  va - the value

   Level: advanced

   Notes:
   The matrix must have been assembled with `MatAssemblyBegin()` and `MatAssemblyEnd()` before this call

   For efficiency one should use `MatGetValues()` and get several values simultaneously.

   See notes for `MatGetValues()`.

.seealso: [](ch_matrices), `Mat`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValue()`, `MatGetValuesLocal()`, `MatGetValues()`
@*/
static inline PetscErrorCode MatGetValue(Mat mat, PetscInt row, PetscInt col, PetscScalar *va)
{
  return MatGetValues(mat, 1, &row, 1, &col, va);
}

/*@
   MatSetValueLocal - Inserts or adds a single value into a matrix, using a local numbering of the nodes.

   Not Collective

   Input Parameters:
+  mat  - the matrix
.  i    - the row location of the entry
.  j    - the column location of the entry
.  va   - the value to insert
-  mode - either `INSERT_VALUES` or `ADD_VALUES`

   Level: intermediate

   Notes:
   For efficiency one should use `MatSetValuesLocal()` and set several values simultaneously.

   See notes for `MatSetValuesLocal()` for additional information on when and how this function can be used.

.seealso: [](ch_matrices), `MatSetValue()`, `MatSetValuesLocal()`
@*/
static inline PetscErrorCode MatSetValueLocal(Mat mat, PetscInt i, PetscInt j, PetscScalar va, InsertMode mode)
{
  return MatSetValuesLocal(mat, 1, &i, 1, &j, &va, mode);
}

/*MC
   MatPreallocateBegin - Begins the block of code that will count the number of nonzeros per
   row in a matrix providing the data that one can use to correctly preallocate the matrix.

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateBegin(MPI_Comm comm, PetscInt nrows, PetscInt ncols, PetscInt *dnz, PetscInt *onz)

   Collective

   Input Parameters:
+  comm  - the communicator that will share the eventually allocated matrix
.  nrows - the number of LOCAL rows in the matrix
-  ncols - the number of LOCAL columns in the matrix

   Output Parameters:
+  dnz - the array that will be passed to the matrix preallocation routines
-  onz - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   This is a macro that handles its own error checking, it does not return an error code.

   Do not malloc or free `dnz` and `onz`, that is handled internally by these routines

   Developer Note:
   This is a MACRO, not a function, because it has a leading { that is closed by `PetscPreallocateFinalize()`.

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`, `MatPreallocateSetLocal()`,
          `MatPreallocateSymmetricSetLocalBlock()`
M*/
#define MatPreallocateBegin(comm, nrows, ncols, dnz, onz) \
  do { \
    PetscInt              __nrows = (nrows), __ncols = (ncols), __rstart, __end = 0; \
    PetscInt PETSC_UNUSED __start; \
    PetscCall(PetscCalloc2(__nrows, &(dnz), __nrows, &(onz))); \
    PetscCallMPI(MPI_Scan(&__ncols, &__end, 1, MPIU_INT, MPI_SUM, comm)); \
    __start = __end - __ncols; \
    (void)__start; \
    PetscCallMPI(MPI_Scan(&__nrows, &__rstart, 1, MPIU_INT, MPI_SUM, comm)); \
  __rstart -= __nrows

#define MatPreallocateInitialize(...) PETSC_DEPRECATED_MACRO(3, 18, 0, "MatPreallocateBegin()", ) MatPreallocateBegin(__VA_ARGS__)

/*MC
   MatPreallocateSetLocal - Indicates the locations (rows and columns) in the matrix where nonzeros will be
   inserted using a local number of the rows and columns

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateSetLocal(ISLocalToGlobalMappping map, PetscInt nrows, PetscInt *rows, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  map   - the row mapping from local numbering to global numbering
.  nrows - the number of rows indicated
.  rows  - the indices of the rows
.  cmap  - the column mapping from local to global numbering
.  ncols - the number of columns in the matrix
.  cols  - the columns indicated
.  dnz   - the array that will be passed to the matrix preallocation routines
-  onz   - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz`, that is handled internally by these routines

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`,
          `MatPreallocateBegin()`, `MatPreallocateSymmetricSetLocalBlock()`, `MatPreallocateSetLocalRemoveDups()`
M*/
#define MatPreallocateSetLocal(rmap, nrows, rows, cmap, ncols, cols, dnz, onz) \
  PetscMacroReturnStandard(PetscCall(ISLocalToGlobalMappingApply(rmap, nrows, rows, rows)); PetscCall(ISLocalToGlobalMappingApply(cmap, ncols, cols, cols)); for (PetscInt __l = 0; __l < nrows; __l++) PetscCall(MatPreallocateSet((rows)[__l], ncols, cols, dnz, onz));)

/*MC
   MatPreallocateSetLocalRemoveDups - Indicates the locations (rows and columns) in the matrix where nonzeros will be
   inserted using a local number of the rows and columns. This version removes any duplicate columns in cols

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateSetLocalRemoveDups(ISLocalToGlobalMappping map, PetscInt nrows, PetscInt *rows, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  map   - the row mapping from local numbering to global numbering
.  nrows - the number of rows indicated
.  rows  - the indices of the rows (these values are mapped to the global values)
.  cmap  - the column mapping from local to global numbering
.  ncols - the number of columns in the matrix   (this value will be changed if duplicate columns are found)
.  cols  - the columns indicated (these values are mapped to the global values, they are then sorted and duplicates removed)
.  dnz   - the array that will be passed to the matrix preallocation routines
-  onz   - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz`, that is handled internally by these routines

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`,
          `MatPreallocateBegin()`, `MatPreallocateSymmetricSetLocalBlock()`, `MatPreallocateSetLocal()`
M*/
#define MatPreallocateSetLocalRemoveDups(rmap, nrows, rows, cmap, ncols, cols, dnz, onz) \
  PetscMacroReturnStandard(PetscCall(ISLocalToGlobalMappingApply(rmap, nrows, rows, rows)); PetscCall(ISLocalToGlobalMappingApply(cmap, ncols, cols, cols)); PetscCall(PetscSortRemoveDupsInt(&ncols, cols)); for (PetscInt __l = 0; __l < nrows; __l++) PetscCall(MatPreallocateSet((rows)[__l], ncols, cols, dnz, onz));)

/*MC
   MatPreallocateSetLocalBlock - Indicates the locations (rows and columns) in the matrix where nonzeros will be
   inserted using a local number of the rows and columns

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateSetLocalBlock(ISLocalToGlobalMappping map, PetscInt nrows, PetscInt *rows, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  map   - the row mapping from local numbering to global numbering
.  nrows - the number of rows indicated
.  rows  - the indices of the rows
.  cmap  - the column mapping from local to global numbering
.  ncols - the number of columns in the matrix
.  cols  - the columns indicated
.  dnz   - the array that will be passed to the matrix preallocation routines
-  onz   - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz`, that is handled internally by these routines

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`,
          `MatPreallocateBegin()`, `MatPreallocateSymmetricSetLocalBlock()`
M*/
#define MatPreallocateSetLocalBlock(rmap, nrows, rows, cmap, ncols, cols, dnz, onz) \
  PetscMacroReturnStandard(PetscCall(ISLocalToGlobalMappingApplyBlock(rmap, nrows, rows, rows)); PetscCall(ISLocalToGlobalMappingApplyBlock(cmap, ncols, cols, cols)); for (PetscInt __l = 0; __l < nrows; __l++) PetscCall(MatPreallocateSet((rows)[__l], ncols, cols, dnz, onz));)

/*MC
   MatPreallocateSymmetricSetLocalBlock - Indicates the locations (rows and columns) in the matrix where nonzeros will be
   inserted using a local number of the rows and columns

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateSymmetricSetLocalBlock(ISLocalToGlobalMappping map, PetscInt nrows, PetscInt *rows, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  map   - the mapping between local numbering and global numbering
.  nrows - the number of rows indicated
.  rows  - the indices of the rows
.  ncols - the number of columns in the matrix
.  cols  - the columns indicated
.  dnz   - the array that will be passed to the matrix preallocation routines
-  onz   - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz` that is handled internally by these routines

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`,
          `MatPreallocateBegin()`, `MatPreallocateSetLocal()`
M*/
#define MatPreallocateSymmetricSetLocalBlock(map, nrows, rows, ncols, cols, dnz, onz) \
  PetscMacroReturnStandard(PetscCall(ISLocalToGlobalMappingApplyBlock(map, nrows, rows, rows)); PetscCall(ISLocalToGlobalMappingApplyBlock(map, ncols, cols, cols)); for (PetscInt __l = 0; __l < nrows; __l++) PetscCall(MatPreallocateSymmetricSetBlock((rows)[__l], ncols, cols, dnz, onz));)

/*MC
   MatPreallocateSet - Indicates the locations (rows and columns) in the matrix where nonzeros will be
   inserted using a local number of the rows and columns

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateSet(PetscInt nrows, PetscInt *rows, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  row   - the row
.  ncols - the number of columns in the matrix
-  cols  - the columns indicated

   Output Parameters:
+  dnz - the array that will be passed to the matrix preallocation routines
-  onz - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz` that is handled internally by these routines

   This is a MACRO, not a function, because it uses variables declared in MatPreallocateBegin().

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`,
          `MatPreallocateBegin()`, `MatPreallocateSetLocal()`
M*/
#define MatPreallocateSet(row, nc, cols, dnz, onz) \
  PetscMacroReturnStandard(PetscCheck(row >= __rstart, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Trying to set preallocation for row %" PetscInt_FMT " less than first local row %" PetscInt_FMT, row, __rstart); PetscCheck(row < __rstart + __nrows, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Trying to set preallocation for row %" PetscInt_FMT " greater than last local row %" PetscInt_FMT, row, __rstart + __nrows - 1); for (PetscInt __i = 0; __i < nc; ++__i) { \
    if ((cols)[__i] < __start || (cols)[__i] >= __end) onz[row - __rstart]++; \
    else if (dnz[row - __rstart] < __ncols) dnz[row - __rstart]++; \
  })

/*MC
   MatPreallocateSymmetricSetBlock - Indicates the locations (rows and columns) in the matrix where nonzeros will be
   inserted using a local number of the rows and columns

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateSymmetricSetBlock(PetscInt nrows, PetscInt *rows, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  nrows - the number of rows indicated
.  rows  - the indices of the rows
.  ncols - the number of columns in the matrix
.  cols  - the columns indicated
.  dnz   - the array that will be passed to the matrix preallocation routines
-  onz   - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz` that is handled internally by these routines

   This is a MACRO, not a function, because it uses variables declared in MatPreallocateBegin().

.seealso: [](ch_matrices), `MatPreallocateEnd()`, `MatPreallocateSet()`, `MatPreallocateBegin()`,
          `MatPreallocateSymmetricSetLocalBlock()`, `MatPreallocateSetLocal()`
M*/
#define MatPreallocateSymmetricSetBlock(row, nc, cols, dnz, onz) \
  PetscMacroReturnStandard(for (PetscInt __i = 0; __i < nc; __i++) { \
    if (cols[__i] >= __end) onz[row - __rstart]++; \
    else if (cols[__i] >= row && dnz[row - __rstart] < __ncols) dnz[row - __rstart]++; \
  })

/*MC
   MatPreallocateLocation -  An alternative to MatPreallocateSet() that puts the nonzero locations into the matrix if it exists

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateLocations(Mat A, PetscInt row, PetscInt ncols, PetscInt *cols, PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  A     - matrix
.  row   - row where values exist (must be local to this process)
.  ncols - number of columns
.  cols  - columns with nonzeros
.  dnz   - the array that will be passed to the matrix preallocation routines
-  onz   - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz` that is handled internally by these routines

   Developer Note:
   This is a MACRO, not a function, because it uses a bunch of variables private to the MatPreallocation.... routines.

.seealso: [](ch_matrices), `MatPreallocateBegin()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`, `MatPreallocateSetLocal()`,
          `MatPreallocateSymmetricSetLocalBlock()`
M*/
#define MatPreallocateLocation(A, row, ncols, cols, dnz, onz) (A ? MatSetValues(A, 1, &row, ncols, cols, NULL, INSERT_VALUES) : MatPreallocateSet(row, ncols, cols, dnz, onz))

/*MC
   MatPreallocateEnd - Ends the block of code that will count the number of nonzeros per
   row in a matrix providing the data that one can use to correctly preallocate the matrix.

   Synopsis:
   #include <petscmat.h>
   PetscErrorCode MatPreallocateEnd(PetscInt *dnz, PetscInt *onz)

   Collective

   Input Parameters:
+  dnz - the array that was be passed to the matrix preallocation routines
-  onz - the other array passed to the matrix preallocation routines

   Level: deprecated (since v3.19)

   Notes:
   This routine is no longer needed since assembling matrices without explicit preallocation will not be slower than
   the use of this routine

   Do not malloc or free `dnz` and `onz`, that is handled internally by these routines

   Developer Note:
   This is a MACRO, not a function, because it closes the { started in MatPreallocateBegin().

.seealso: [](ch_matrices), `MatPreallocateBegin()`, `MatPreallocateSet()`, `MatPreallocateSymmetricSetBlock()`, `MatPreallocateSetLocal()`,
          `MatPreallocateSymmetricSetLocalBlock()`
M*/
#define MatPreallocateEnd(dnz, onz) \
  PetscCall(PetscFree2(dnz, onz)); \
  } \
  while (0)

#define MatPreallocateFinalize(...) PETSC_DEPRECATED_MACRO(3, 18, 0, "MatPreallocateEnd()", ) MatPreallocateEnd(__VA_ARGS__)

/* Routines unique to particular data structures */
PETSC_EXTERN PetscErrorCode MatShellGetContext(Mat, PetscCtxRt);

PETSC_EXTERN PetscErrorCode MatInodeAdjustForInodes(Mat, IS *, IS *);
PETSC_EXTERN PetscErrorCode MatInodeGetInodeSizes(Mat, PetscInt *, PetscInt *[], PetscInt *);

PETSC_EXTERN PetscErrorCode MatSeqAIJSetColumnIndices(Mat, PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqBAIJSetColumnIndices(Mat, PetscInt[]);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqBAIJWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqSBAIJWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJFromTriple(MPI_Comm, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscScalar[], Mat *, PetscCount, PetscBool);

#define MAT_SKIP_ALLOCATION -4

PETSC_EXTERN PetscErrorCode MatSeqBAIJSetPreallocation(Mat, PetscInt, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqSBAIJSetPreallocation(Mat, PetscInt, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJSetPreallocation(Mat, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJSetTotalPreallocation(Mat, PetscInt);

PETSC_EXTERN PetscErrorCode MatMPIBAIJSetPreallocation(Mat, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPISBAIJSetPreallocation(Mat, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPIAIJSetPreallocation(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJSetPreallocationCSR(Mat, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSeqBAIJSetPreallocationCSR(Mat, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIAIJSetPreallocationCSR(Mat, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIBAIJSetPreallocationCSR(Mat, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIAdjSetPreallocation(Mat, PetscInt[], PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPIAdjToSeq(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatMPIAdjToSeqRankZero(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatMPIDenseSetPreallocation(Mat, PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSeqDenseSetPreallocation(Mat, PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetSeqAIJ(Mat, Mat *, Mat *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatMPIBAIJGetSeqBAIJ(Mat, Mat *, Mat *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatMPIAdjCreateNonemptySubcommMat(Mat, Mat *);

PETSC_EXTERN PetscErrorCode MatDenseGetLDA(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatDenseSetLDA(Mat, PetscInt);
PETSC_DEPRECATED_FUNCTION(3, 14, 0, "MatDenseSetLDA()", ) static inline PetscErrorCode MatSeqDenseSetLDA(Mat A, PetscInt lda)
{
  return MatDenseSetLDA(A, lda);
}
PETSC_EXTERN PetscErrorCode MatDenseGetLocalMatrix(Mat, Mat *);

PETSC_EXTERN PetscErrorCode MatBlockMatSetPreallocation(Mat, PetscInt, PetscInt, const PetscInt[]);

PETSC_EXTERN PetscErrorCode MatStoreValues(Mat);
PETSC_EXTERN PetscErrorCode MatRetrieveValues(Mat);

PETSC_EXTERN PetscErrorCode MatFindNonzeroRows(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatFindZeroRows(Mat, IS *);
/*
  These routines are not usually accessed directly, rather solving is
  done through the KSP and PC interfaces.
*/

PETSC_EXTERN PetscErrorCode    MatGetOrdering(Mat, MatOrderingType, IS *, IS *);
PETSC_EXTERN PetscErrorCode    MatGetOrderingList(PetscFunctionList *);
PETSC_EXTERN PetscErrorCode    MatOrderingRegister(const char[], PetscErrorCode (*)(Mat, MatOrderingType, IS *, IS *));
PETSC_EXTERN PetscFunctionList MatOrderingList;

#include "petscmatcoarsen.h"

PETSC_EXTERN PetscErrorCode MatReorderForNonzeroDiagonal(Mat, PetscReal, IS, IS);
PETSC_EXTERN PetscErrorCode MatCreateLaplacian(Mat, PetscReal, PetscBool, Mat *);

PETSC_EXTERN PetscErrorCode MatFactorGetPreferredOrdering(Mat, MatFactorType, MatOrderingType *);

PETSC_EXTERN PetscErrorCode MatFactorGetError(Mat, MatFactorError *);
PETSC_EXTERN PetscErrorCode MatFactorClearError(Mat);
PETSC_EXTERN PetscErrorCode MatFactorGetErrorZeroPivot(Mat, PetscReal *, PetscInt *);

PETSC_EXTERN PetscErrorCode MatFactorInfoInitialize(MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatCholeskyFactor(Mat, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatCholeskyFactorSymbolic(Mat, Mat, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatCholeskyFactorNumeric(Mat, Mat, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatLUFactor(Mat, IS, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatILUFactor(Mat, IS, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatLUFactorSymbolic(Mat, Mat, IS, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatILUFactorSymbolic(Mat, Mat, IS, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatICCFactorSymbolic(Mat, Mat, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatICCFactor(Mat, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatLUFactorNumeric(Mat, Mat, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatQRFactor(Mat, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatQRFactorSymbolic(Mat, Mat, IS, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatQRFactorNumeric(Mat, Mat, const MatFactorInfo *);
PETSC_EXTERN PetscErrorCode MatGetInertia(Mat, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatSolve(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatForwardSolve(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatBackwardSolve(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatSolveAdd(Mat, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatSolveTranspose(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatSolveTransposeAdd(Mat, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatSolves(Mat, Vecs, Vecs);
PETSC_EXTERN PetscErrorCode MatSetUnfactored(Mat);

PETSC_EXTERN PetscErrorCode MatFactorSetSchurIS(Mat, IS);
PETSC_EXTERN PetscErrorCode MatFactorGetSchurComplement(Mat, Mat *, MatFactorSchurStatus *);
PETSC_EXTERN PetscErrorCode MatFactorRestoreSchurComplement(Mat, Mat *, MatFactorSchurStatus);
PETSC_EXTERN PetscErrorCode MatFactorInvertSchurComplement(Mat);
PETSC_EXTERN PetscErrorCode MatFactorCreateSchurComplement(Mat, Mat *, MatFactorSchurStatus *);
PETSC_EXTERN PetscErrorCode MatFactorSolveSchurComplement(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatFactorSolveSchurComplementTranspose(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatFactorFactorizeSchurComplement(Mat);

PETSC_EXTERN PetscErrorCode MatSeqDenseInvert(Mat);
PETSC_EXTERN PetscErrorCode MatSOR(Mat, Vec, PetscReal, MatSORType, PetscReal, PetscInt, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode MatColoringCreate(Mat, MatColoring *);
PETSC_EXTERN PetscErrorCode MatColoringGetDegrees(Mat, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode MatColoringDestroy(MatColoring *);
PETSC_EXTERN PetscErrorCode MatColoringView(MatColoring, PetscViewer);
PETSC_EXTERN PetscErrorCode MatColoringSetType(MatColoring, MatColoringType);
PETSC_EXTERN PetscErrorCode MatColoringSetFromOptions(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringSetDistance(MatColoring, PetscInt);
PETSC_EXTERN PetscErrorCode MatColoringGetDistance(MatColoring, PetscInt *);
PETSC_EXTERN PetscErrorCode MatColoringSetMaxColors(MatColoring, PetscInt);
PETSC_EXTERN PetscErrorCode MatColoringGetMaxColors(MatColoring, PetscInt *);
PETSC_EXTERN PetscErrorCode MatColoringApply(MatColoring, ISColoring *);
PETSC_EXTERN PetscErrorCode MatColoringRegister(const char[], PetscErrorCode (*)(MatColoring));
PETSC_EXTERN PetscErrorCode MatColoringPatch(Mat, PetscInt, PetscInt, ISColoringValue[], ISColoring *);
PETSC_EXTERN PetscErrorCode MatColoringSetWeightType(MatColoring, MatColoringWeightType);
PETSC_EXTERN PetscErrorCode MatColoringSetWeights(MatColoring, PetscReal *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatColoringCreateWeights(MatColoring, PetscReal **, PetscInt **lperm);
PETSC_EXTERN PetscErrorCode MatColoringTest(MatColoring, ISColoring);
PETSC_DEPRECATED_FUNCTION(3, 10, 0, "MatColoringTest()", ) static inline PetscErrorCode MatColoringTestValid(MatColoring matcoloring, ISColoring iscoloring)
{
  return MatColoringTest(matcoloring, iscoloring);
}
PETSC_EXTERN PetscErrorCode MatISColoringTest(Mat, ISColoring);

PETSC_EXTERN PetscErrorCode MatFDColoringCreate(Mat, ISColoring, MatFDColoring *);
PETSC_EXTERN PetscErrorCode MatFDColoringDestroy(MatFDColoring *);
PETSC_EXTERN PetscErrorCode MatFDColoringView(MatFDColoring, PetscViewer);
PETSC_EXTERN PetscErrorCode MatFDColoringSetFunction(MatFDColoring, MatFDColoringFn *, void *);
PETSC_EXTERN PetscErrorCode MatFDColoringGetFunction(MatFDColoring, MatFDColoringFn **, void **);
PETSC_EXTERN PetscErrorCode MatFDColoringSetParameters(MatFDColoring, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode MatFDColoringSetFromOptions(MatFDColoring);
PETSC_EXTERN PetscErrorCode MatFDColoringApply(Mat, MatFDColoring, Vec, void *);
PETSC_EXTERN PetscErrorCode MatFDColoringSetF(MatFDColoring, Vec);
PETSC_EXTERN PetscErrorCode MatFDColoringGetPerturbedColumns(MatFDColoring, PetscInt *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatFDColoringSetUp(Mat, ISColoring, MatFDColoring);
PETSC_EXTERN PetscErrorCode MatFDColoringSetBlockSize(MatFDColoring, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode MatFDColoringSetValues(Mat, MatFDColoring, const PetscScalar *);

PETSC_EXTERN PetscErrorCode MatTransposeColoringCreate(Mat, ISColoring, MatTransposeColoring *);
PETSC_EXTERN PetscErrorCode MatTransColoringApplySpToDen(MatTransposeColoring, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatTransColoringApplyDenToSp(MatTransposeColoring, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatTransposeColoringDestroy(MatTransposeColoring *);

PETSC_EXTERN PetscErrorCode MatPartitioningCreate(MPI_Comm, MatPartitioning *);
PETSC_EXTERN PetscErrorCode MatPartitioningSetType(MatPartitioning, MatPartitioningType);
PETSC_EXTERN PetscErrorCode MatPartitioningSetNParts(MatPartitioning, PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningSetAdjacency(MatPartitioning, Mat);
PETSC_EXTERN PetscErrorCode MatPartitioningSetNumberVertexWeights(MatPartitioning, PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningSetVertexWeights(MatPartitioning, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatPartitioningSetPartitionWeights(MatPartitioning, const PetscReal[]);
PETSC_EXTERN PetscErrorCode MatPartitioningSetUseEdgeWeights(MatPartitioning, PetscBool);
PETSC_EXTERN PetscErrorCode MatPartitioningGetUseEdgeWeights(MatPartitioning, PetscBool *);
PETSC_EXTERN PetscErrorCode MatPartitioningApply(MatPartitioning, IS *);
PETSC_EXTERN PetscErrorCode MatPartitioningImprove(MatPartitioning, IS *);
PETSC_EXTERN PetscErrorCode MatPartitioningViewImbalance(MatPartitioning, IS);
PETSC_EXTERN PetscErrorCode MatPartitioningApplyND(MatPartitioning, IS *);
PETSC_EXTERN PetscErrorCode MatPartitioningDestroy(MatPartitioning *);
PETSC_EXTERN PetscErrorCode MatPartitioningRegister(const char[], PetscErrorCode (*)(MatPartitioning));
PETSC_EXTERN PetscErrorCode MatPartitioningView(MatPartitioning, PetscViewer);
PETSC_EXTERN PetscErrorCode MatPartitioningViewFromOptions(MatPartitioning, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode MatPartitioningSetFromOptions(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningGetType(MatPartitioning, MatPartitioningType *);

PETSC_EXTERN PetscErrorCode MatPartitioningParmetisSetRepartition(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningParmetisSetCoarseSequential(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningParmetisGetEdgeCut(MatPartitioning, PetscInt *);

PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetGlobal(MatPartitioning, MPChacoGlobalType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetGlobal(MatPartitioning, MPChacoGlobalType *);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetLocal(MatPartitioning, MPChacoLocalType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetLocal(MatPartitioning, MPChacoLocalType *);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetCoarseLevel(MatPartitioning, PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenSolver(MatPartitioning, MPChacoEigenType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenSolver(MatPartitioning, MPChacoEigenType *);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenTol(MatPartitioning, PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenTol(MatPartitioning, PetscReal *);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenNumber(MatPartitioning, PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenNumber(MatPartitioning, PetscInt *);

#define MP_PARTY_OPT "opt"
#define MP_PARTY_LIN "lin"
#define MP_PARTY_SCA "sca"
#define MP_PARTY_RAN "ran"
#define MP_PARTY_GBF "gbf"
#define MP_PARTY_GCF "gcf"
#define MP_PARTY_BUB "bub"
#define MP_PARTY_DEF "def"
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetGlobal(MatPartitioning, const char *);
#define MP_PARTY_HELPFUL_SETS  "hs"
#define MP_PARTY_KERNIGHAN_LIN "kl"
#define MP_PARTY_NONE          "no"
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetLocal(MatPartitioning, const char *);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetCoarseLevel(MatPartitioning, PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetBipart(MatPartitioning, PetscBool);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetMatchOptimization(MatPartitioning, PetscBool);

PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchSetImbalance(MatPartitioning, PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchGetImbalance(MatPartitioning, PetscReal *);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchSetStrategy(MatPartitioning, MPPTScotchStrategyType);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchGetStrategy(MatPartitioning, MPPTScotchStrategyType *);

/*
 * hierarchical partitioning
 */
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalGetFineparts(MatPartitioning, IS *);
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalGetCoarseparts(MatPartitioning, IS *);
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalSetNcoarseparts(MatPartitioning, PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningHierarchicalSetNfineparts(MatPartitioning, PetscInt);

PETSC_EXTERN PetscErrorCode MatMeshToCellGraph(Mat, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatMeshToCellGraphRegister(const char[], PetscErrorCode (*)(Mat, PetscInt, Mat *));

PETSC_EXTERN PetscErrorCode MatSetOperation(Mat, MatOperation, PetscErrorCodeFn *);
PETSC_EXTERN PetscErrorCode MatGetOperation(Mat, MatOperation, PetscErrorCodeFn **);
PETSC_EXTERN PetscErrorCode MatHasOperation(Mat, MatOperation, PetscBool *);
PETSC_EXTERN PetscErrorCode MatHasCongruentLayouts(Mat, PetscBool *);
PETSC_DEPRECATED_FUNCTION(3, 14, 0, "MatProductClear()", ) static inline PetscErrorCode MatFreeIntermediateDataStructures(Mat A)
{
  return MatProductClear(A);
}
PETSC_EXTERN PetscErrorCode MatShellSetOperation(Mat, MatOperation, PetscErrorCodeFn *);
PETSC_EXTERN PetscErrorCode MatShellGetOperation(Mat, MatOperation, PetscErrorCodeFn **);
PETSC_EXTERN PetscErrorCode MatShellSetContext(Mat, PetscCtx);
PETSC_EXTERN PetscErrorCode MatShellSetContextDestroy(Mat, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode MatShellSetVecType(Mat, VecType);
PETSC_EXTERN PetscErrorCode MatShellTestMult(Mat, PetscErrorCode (*)(void *, Vec, Vec), Vec, void *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatShellTestMultTranspose(Mat, PetscErrorCode (*)(void *, Vec, Vec), Vec, void *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatShellSetManageScalingShifts(Mat);
PETSC_EXTERN PetscErrorCode MatShellSetMatProductOperation(Mat, MatProductType, PetscErrorCode (*)(Mat, Mat, Mat, void **), PetscErrorCode (*)(Mat, Mat, Mat, void *), PetscCtxDestroyFn *, MatType, MatType);
PETSC_EXTERN PetscErrorCode MatIsShell(Mat, PetscBool *);

/*
   Codes for matrices stored on disk. By default they are
   stored in a universal format. By changing the format with
   PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE); the matrices will
   be stored in a way natural for the matrix, for example dense matrices
   would be stored as dense. Matrices stored this way may only be
   read into matrices of the same type.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

PETSC_EXTERN PetscErrorCode MatMPIBAIJSetHashTableFactor(Mat, PetscReal);

PETSC_EXTERN PetscErrorCode MatISSetLocalMatType(Mat, MatType);
PETSC_EXTERN PetscErrorCode MatISSetPreallocation(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatISSetAllowRepeated(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatISGetAllowRepeated(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatISStoreL2L(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatISFixLocalEmpty(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatISGetLocalMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatISRestoreLocalMat(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatISSetLocalMat(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatISGetLocalToGlobalMapping(Mat, ISLocalToGlobalMapping *, ISLocalToGlobalMapping *);

PETSC_EXTERN PetscErrorCode MatNullSpaceCreate(MPI_Comm, PetscBool, PetscInt, const Vec[], MatNullSpace *);
PETSC_EXTERN PetscErrorCode MatNullSpaceSetFunction(MatNullSpace, MatNullSpaceRemoveFn *, void *);
PETSC_EXTERN PetscErrorCode MatNullSpaceDestroy(MatNullSpace *);
PETSC_EXTERN PetscErrorCode MatNullSpaceRemove(MatNullSpace, Vec);
PETSC_EXTERN PetscErrorCode MatGetNullSpace(Mat, MatNullSpace *);
PETSC_EXTERN PetscErrorCode MatGetTransposeNullSpace(Mat, MatNullSpace *);
PETSC_EXTERN PetscErrorCode MatSetTransposeNullSpace(Mat, MatNullSpace);
PETSC_EXTERN PetscErrorCode MatSetNullSpace(Mat, MatNullSpace);
PETSC_EXTERN PetscErrorCode MatSetNearNullSpace(Mat, MatNullSpace);
PETSC_EXTERN PetscErrorCode MatGetNearNullSpace(Mat, MatNullSpace *);
PETSC_EXTERN PetscErrorCode MatGetNullSpaces(PetscInt, Mat[], MatNullSpace *[]);
PETSC_EXTERN PetscErrorCode MatRestoreNullSpaces(PetscInt, Mat[], MatNullSpace *[]);
PETSC_EXTERN PetscErrorCode MatNullSpaceTest(MatNullSpace, Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatNullSpaceView(MatNullSpace, PetscViewer);
PETSC_EXTERN PetscErrorCode MatNullSpaceGetVecs(MatNullSpace, PetscBool *, PetscInt *, const Vec **);
PETSC_EXTERN PetscErrorCode MatNullSpaceCreateRigidBody(Vec, MatNullSpace *);

PETSC_EXTERN PetscErrorCode MatReorderingSeqSBAIJ(Mat, IS);
PETSC_EXTERN PetscErrorCode MatSeqSBAIJSetColumnIndices(Mat, PetscInt *);

PETSC_EXTERN PetscErrorCode MatCreateMAIJ(Mat, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatMAIJRedimension(Mat, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatMAIJGetAIJ(Mat, Mat *);

PETSC_EXTERN PetscErrorCode MatComputeOperator(Mat, MatType, Mat *);
PETSC_EXTERN PetscErrorCode MatComputeOperatorTranspose(Mat, MatType, Mat *);

PETSC_DEPRECATED_FUNCTION(3, 12, 0, "MatComputeOperator()", ) static inline PetscErrorCode MatComputeExplicitOperator(Mat A, Mat *B)
{
  return MatComputeOperator(A, PETSC_NULLPTR, B);
}
PETSC_DEPRECATED_FUNCTION(3, 12, 0, "MatComputeOperatorTranspose()", ) static inline PetscErrorCode MatComputeExplicitOperatorTranspose(Mat A, Mat *B)
{
  return MatComputeOperatorTranspose(A, PETSC_NULLPTR, B);
}

PETSC_EXTERN PetscErrorCode MatCreateKAIJ(Mat, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatKAIJGetAIJ(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatKAIJGetS(Mat, PetscInt *, PetscInt *, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJGetSRead(Mat, PetscInt *, PetscInt *, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJRestoreS(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJRestoreSRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJGetT(Mat, PetscInt *, PetscInt *, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJGetTRead(Mat, PetscInt *, PetscInt *, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJRestoreT(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJRestoreTRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatKAIJSetAIJ(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatKAIJSetS(Mat, PetscInt, PetscInt, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatKAIJSetT(Mat, PetscInt, PetscInt, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatKAIJGetScaledIdentity(Mat, PetscBool *);

PETSC_EXTERN PetscErrorCode MatDiagonalScaleLocal(Mat, Vec);

PETSC_EXTERN PetscErrorCode MatMFFDInitializePackage(void);
PETSC_EXTERN PetscErrorCode MatMFFDFinalizePackage(void);

PETSC_EXTERN PetscErrorCode MatCreateMFFD(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatMFFDSetBase(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunction(Mat, MatMFFDFn *, void *);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunctioni(Mat, MatMFFDiFn *);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunctioniBase(Mat, MatMFFDiBaseFn *);
PETSC_EXTERN PetscErrorCode MatMFFDSetHHistory(Mat, PetscScalar[], PetscInt);
PETSC_EXTERN PetscErrorCode MatMFFDResetHHistory(Mat);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunctionError(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatMFFDSetPeriod(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatMFFDGetH(Mat, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatMFFDSetOptionsPrefix(Mat, const char[]);
PETSC_EXTERN PetscErrorCode MatMFFDCheckPositivity(void *, Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatMFFDSetCheckh(Mat, MatMFFDCheckhFn *, void *);

PETSC_EXTERN PetscErrorCode MatMFFDSetType(Mat, MatMFFDType);
PETSC_EXTERN PetscErrorCode MatMFFDRegister(const char[], PetscErrorCode (*)(MatMFFD));

PETSC_EXTERN PetscErrorCode MatMFFDDSSetUmin(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatMFFDWPSetComputeNormU(Mat, PetscBool);

PETSC_EXTERN PetscErrorCode MatFDColoringSetType(MatFDColoring, MatMFFDType);

PETSC_EXTERN PetscErrorCode PetscViewerMathematicaPutMatrix(PetscViewer, PetscInt, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscViewerMathematicaPutCSRMatrix(PetscViewer, PetscInt, PetscInt, PetscInt *, PetscInt *, PetscReal *);

#if PetscDefined(HAVE_H2OPUS)
PETSC_EXTERN PetscErrorCode MatCreateH2OpusFromKernel(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscReal[], PetscBool, MatH2OpusKernelFn *, void *, PetscReal, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateH2OpusFromMat(Mat, PetscInt, const PetscReal[], PetscBool, PetscReal, PetscInt, PetscInt, PetscInt, PetscReal, Mat *);
PETSC_EXTERN PetscErrorCode MatH2OpusSetSamplingMat(Mat, Mat, PetscInt, PetscReal);
PETSC_EXTERN PetscErrorCode MatH2OpusOrthogonalize(Mat);
PETSC_EXTERN PetscErrorCode MatH2OpusCompress(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatH2OpusSetNativeMult(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatH2OpusGetNativeMult(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatH2OpusGetIndexMap(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatH2OpusMapVec(Mat, PetscBool, Vec, Vec *);
PETSC_EXTERN PetscErrorCode MatH2OpusLowRankUpdate(Mat, Mat, Mat, PetscScalar);
#endif

PETSC_EXTERN PetscErrorCode MatCreateHtoolFromKernel(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscReal[], const PetscReal[], MatHtoolKernelFn *, void *, Mat *);
PETSC_EXTERN PetscErrorCode MatHtoolSetKernel(Mat, MatHtoolKernelFn *, void *);
PETSC_EXTERN PetscErrorCode MatHtoolGetPermutationSource(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatHtoolGetPermutationTarget(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatHtoolUsePermutation(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatHtoolUseRecompression(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatHtoolGetEpsilon(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatHtoolSetEpsilon(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatHtoolGetEta(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatHtoolSetEta(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatHtoolGetMaxClusterLeafSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatHtoolSetMaxClusterLeafSize(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatHtoolGetMinTargetDepth(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatHtoolSetMinTargetDepth(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatHtoolGetMinSourceDepth(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatHtoolSetMinSourceDepth(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatHtoolGetBlockTreeConsistency(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatHtoolSetBlockTreeConsistency(Mat, PetscBool);

PETSC_EXTERN PetscErrorCode MatHtoolGetCompressorType(Mat, MatHtoolCompressorType *);
PETSC_EXTERN PetscErrorCode MatHtoolSetCompressorType(Mat, MatHtoolCompressorType);
PETSC_EXTERN PetscErrorCode MatHtoolGetClusteringType(Mat, MatHtoolClusteringType *);
PETSC_EXTERN PetscErrorCode MatHtoolSetClusteringType(Mat, MatHtoolClusteringType);

PETSC_EXTERN PetscErrorCode MatMumpsSetIcntl(Mat, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode MatMumpsGetIcntl(Mat, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode MatMumpsSetCntl(Mat, PetscInt, PetscReal);
PETSC_EXTERN PetscErrorCode MatMumpsGetCntl(Mat, PetscInt, PetscReal *);

PETSC_EXTERN PetscErrorCode MatMumpsGetInfo(Mat, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode MatMumpsGetInfog(Mat, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode MatMumpsGetRinfo(Mat, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode MatMumpsGetRinfog(Mat, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode MatMumpsGetNullPivots(Mat, PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode MatMumpsGetInverse(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatMumpsGetInverseTranspose(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatMumpsSetBlk(Mat, PetscInt, const PetscInt[], const PetscInt[]);

PETSC_EXTERN PetscErrorCode MatMumpsSetOocTmpDir(Mat, const char *);
PETSC_EXTERN PetscErrorCode MatMumpsGetOocTmpDir(Mat, const char *[]);

#if PetscDefined(HAVE_MKL_PARDISO)
PETSC_EXTERN PetscErrorCode MatMkl_PardisoSetCntl(Mat, PetscInt, PetscInt);
#endif

#if PetscDefined(HAVE_MKL_CPARDISO)
PETSC_EXTERN PetscErrorCode MatMkl_CPardisoSetCntl(Mat, PetscInt, PetscInt);
#endif

#if PetscDefined(HAVE_SUPERLU)
PETSC_EXTERN PetscErrorCode MatSuperluSetILUDropTol(Mat, PetscReal);
#endif

#if PetscDefined(HAVE_SUPERLU_DIST)
PETSC_EXTERN PetscErrorCode MatSuperluDistGetDiagU(Mat, PetscScalar *);
#endif

PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetReordering(Mat, MatSTRUMPACKReordering);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetReordering(Mat, MatSTRUMPACKReordering *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetGeometricNxyz(Mat, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetGeometricComponents(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetGeometricWidth(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetColPerm(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetColPerm(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetGPU(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetGPU(Mat, PetscBool *);

PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompression(Mat, MatSTRUMPACKCompressionType);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompression(Mat, MatSTRUMPACKCompressionType *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompRelTol(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompRelTol(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompAbsTol(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompAbsTol(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompMinSepSize(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompMinSepSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompLeafSize(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompLeafSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompLossyPrecision(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompLossyPrecision(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKSetCompButterflyLevels(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatSTRUMPACKGetCompButterflyLevels(Mat, PetscInt *);

PETSC_EXTERN PetscErrorCode MatBindToCPU(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatBoundToCPU(Mat, PetscBool *);
PETSC_DEPRECATED_FUNCTION(3, 13, 0, "MatBindToCPU()", ) static inline PetscErrorCode MatPinToCPU(Mat A, PetscBool flg)
{
  return MatBindToCPU(A, flg);
}
PETSC_EXTERN PetscErrorCode MatSetBindingPropagates(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatGetBindingPropagates(Mat, PetscBool *);

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJCUSPARSE(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateAIJCUSPARSE(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCUSPARSESetFormat(Mat, MatCUSPARSEFormatOperation, MatCUSPARSEStorageFormat);
PETSC_EXTERN PetscErrorCode MatCUSPARSESetUseCPUSolve(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSEGetIJ(Mat, PetscBool, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSERestoreIJ(Mat, PetscBool, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSEGetArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSERestoreArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSEGetArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSERestoreArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSEGetArray(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJCUSPARSERestoreArray(Mat, PetscScalar **);

PETSC_EXTERN PetscErrorCode MatCreateDenseCUDA(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqDenseCUDA(MPI_Comm, PetscInt, PetscInt, PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatDenseCUDAGetArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseCUDAGetArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseCUDAGetArray(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseCUDARestoreArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseCUDARestoreArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseCUDARestoreArray(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseCUDAPlaceArray(Mat, const PetscScalar *);
PETSC_EXTERN PetscErrorCode MatDenseCUDAReplaceArray(Mat, const PetscScalar *);
PETSC_EXTERN PetscErrorCode MatDenseCUDAResetArray(Mat);
PETSC_EXTERN PetscErrorCode MatDenseCUDASetPreallocation(Mat, PetscScalar *);

PETSC_EXTERN PetscErrorCode MatCreateSELLCUDA(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
#endif

#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJHIPSPARSE(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateAIJHIPSPARSE(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatHIPSPARSESetFormat(Mat, MatHIPSPARSEFormatOperation, MatHIPSPARSEStorageFormat);
PETSC_EXTERN PetscErrorCode MatHIPSPARSESetUseCPUSolve(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSEGetIJ(Mat, PetscBool, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreIJ(Mat, PetscBool, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSEGetArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSEGetArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSEGetArray(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreArray(Mat, PetscScalar **);

PETSC_EXTERN PetscErrorCode MatCreateDenseHIP(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqDenseHIP(MPI_Comm, PetscInt, PetscInt, PetscScalar[], Mat *);
PETSC_EXTERN PetscErrorCode MatDenseHIPGetArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseHIPGetArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseHIPGetArray(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseHIPRestoreArrayWrite(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseHIPRestoreArrayRead(Mat, const PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseHIPRestoreArray(Mat, PetscScalar **);
PETSC_EXTERN PetscErrorCode MatDenseHIPPlaceArray(Mat, const PetscScalar *);
PETSC_EXTERN PetscErrorCode MatDenseHIPReplaceArray(Mat, const PetscScalar *);
PETSC_EXTERN PetscErrorCode MatDenseHIPResetArray(Mat);
PETSC_EXTERN PetscErrorCode MatDenseHIPSetPreallocation(Mat, PetscScalar *);
PETSC_EXTERN PetscErrorCode MatCreateSELLHIP(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
#endif

#if PetscDefined(HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJViennaCL(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateAIJViennaCL(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
#endif

#if PetscDefined(HAVE_KOKKOS)
PETSC_EXTERN PetscErrorCode MatCreateAIJKokkos(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt, const PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJKokkos(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Mat *);
#endif

#if PetscDefined(HAVE_FFTW)
PETSC_EXTERN PetscErrorCode VecScatterPetscToFFTW(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecScatterFFTWToPetsc(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatCreateVecsFFTW(Mat, Vec *, Vec *, Vec *);
#endif

#if PetscDefined(HAVE_SCALAPACK) && (PetscDefined(USE_REAL_SINGLE) || PetscDefined(USE_REAL_DOUBLE))
PETSC_EXTERN PetscErrorCode MatCreateScaLAPACK(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatScaLAPACKSetBlockSizes(Mat, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode MatScaLAPACKGetBlockSizes(Mat, PetscInt *, PetscInt *);
#endif

PETSC_EXTERN PetscErrorCode MatCreateNest(MPI_Comm, PetscInt, const IS[], PetscInt, const IS[], const Mat[], Mat *);
PETSC_EXTERN PetscErrorCode MatNestGetSize(Mat, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode MatNestGetISs(Mat, IS[], IS[]);
PETSC_EXTERN PetscErrorCode MatNestGetLocalISs(Mat, IS[], IS[]);
PETSC_EXTERN PetscErrorCode MatNestGetSubMats(Mat, PetscInt *, PetscInt *, Mat ***);
PETSC_EXTERN PetscErrorCode MatNestGetSubMat(Mat, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatNestSetVecType(Mat, VecType);
PETSC_EXTERN PetscErrorCode MatNestSetSubMats(Mat, PetscInt, const IS[], PetscInt, const IS[], const Mat[]);
PETSC_EXTERN PetscErrorCode MatNestSetSubMat(Mat, PetscInt, PetscInt, Mat);

PETSC_EXTERN PetscErrorCode MatFilter(Mat, PetscReal, PetscBool, PetscBool);
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "MatFilter()", ) static inline PetscErrorCode MatChop(Mat A, PetscReal tol)
{
  return MatFilter(A, tol, PETSC_FALSE, PETSC_FALSE);
}
PETSC_EXTERN PetscErrorCode MatComputeBandwidth(Mat, PetscReal, PetscInt *);

PETSC_EXTERN PetscErrorCode MatSubdomainsCreateCoalesce(Mat, PetscInt, PetscInt *, IS **);

PETSC_EXTERN PetscErrorCode MatPreallocatorPreallocate(Mat, PetscBool, Mat);

PETSC_EXTERN PetscErrorCode MatHeaderMerge(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatHeaderReplace(Mat, Mat *);

PETSC_EXTERN PetscErrorCode MatSeqAIJGetCSRAndMemType(Mat, const PetscInt **, const PetscInt **, PetscScalar **, PetscMemType *);

PETSC_EXTERN PetscErrorCode MatCreateGraph(Mat, PetscBool, PetscBool, PetscReal, PetscInt, PetscInt[], Mat *);
PETSC_EXTERN PetscErrorCode MatEliminateZeros(Mat, PetscBool);

PETSC_EXTERN PetscErrorCode MatCreateDenseFromVecType(MPI_Comm, VecType, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar *, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateDenseWithMemType(MPI_Comm, PetscMemType, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar *, Mat *);

PETSC_EXTERN PetscErrorCode MatSetHPL(Mat, int);
#define PETSCBMHPL "hpl"

PETSC_EXTERN PetscErrorCode MatDFischer(Mat, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec);
