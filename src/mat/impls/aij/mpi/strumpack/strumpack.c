#include <../src/mat/impls/aij/seq/aij.h>            /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <StrumpackSparseSolver.h>

static PetscErrorCode MatGetDiagonal_STRUMPACK(Mat A,Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK factor");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_STRUMPACK(Mat A)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  PetscBool              flg;

  PetscFunctionBegin;
  /* Deallocate STRUMPACK storage */
  PetscStackCall("STRUMPACK_destroy",STRUMPACK_destroy(S));
  PetscCall(PetscFree(A->spptr));;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&flg));
  if (flg) PetscCall(MatDestroy_SeqAIJ(A));
  else PetscCall(MatDestroy_MPIAIJ(A));

  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetReordering_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetColPerm_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetGeometricNxyz_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetGeometricComponents_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetGeometricWidth_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetGPU_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompression_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompRelTol_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompAbsTol_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompMaxRank_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompLeafSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompMinSepSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompLossyPrecision_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetCompButterflyLevels_C",NULL));

  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetReordering_STRUMPACK(Mat F,MatSTRUMPACKReordering reordering)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_reordering_method",STRUMPACK_set_reordering_method(*S,(STRUMPACK_REORDERING_STRATEGY)reordering));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetReordering - Set STRUMPACK fill-reducing reordering

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  reordering - the code to be used to find the fill-reducing reordering
      Possible values: NATURAL=0 METIS=1 PARMETIS=2 SCOTCH=3 PTSCOTCH=4 RCM=5 GEOMETRIC=6

  Options Database:
.   -mat_strumpack_reordering <METIS>  - Sparsity reducing matrix reordering (choose one of) NATURAL METIS PARMETIS SCOTCH PTSCOTCH RCM (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetReordering(Mat F,MatSTRUMPACKReordering reordering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(F,reordering,2);
  PetscTryMethod(F,"MatSTRUMPACKSetReordering_C",(Mat,MatSTRUMPACKReordering),(F,reordering));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetColPerm_STRUMPACK(Mat F,PetscBool cperm)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_matching",STRUMPACK_set_matching(*S,cperm ? STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING : STRUMPACK_MATCHING_NONE));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetColPerm - Set whether STRUMPACK should try to permute the columns of the matrix in order to get a nonzero diagonal

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  cperm - PETSC_TRUE to permute (internally) the columns of the matrix

  Options Database:
.   -mat_strumpack_colperm <cperm> - true to use the permutation

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetColPerm(Mat F,PetscBool cperm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(F,cperm,2);
  PetscTryMethod(F,"MatSTRUMPACKSetColPerm_C",(Mat,PetscBool),(F,cperm));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetGPU_STRUMPACK(Mat F,PetscBool gpu)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  if (gpu) PetscStackCall("STRUMPACK_enable_gpu",STRUMPACK_enable_gpu(*S));
  else PetscStackCall("STRUMPACK_disable_gpu",STRUMPACK_disable_gpu(*S));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetGPU - Set whether STRUMPACK should enable GPU acceleration (not supported for all compression types)
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  gpu - whether or not to use GPU acceleration

  Options Database:
.   -mat_strumpack_gpu <gpu>

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetGPU(Mat F,PetscBool gpu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(F,gpu,2);
  PetscTryMethod(F,"MatSTRUMPACKSetGPU_C",(Mat,PetscBool),(F,gpu));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompression_STRUMPACK(Mat F,MatSTRUMPACKCompression comp)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression",STRUMPACK_set_compression(*S,(STRUMPACK_COMPRESSION_TYPE)comp));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompression - Set STRUMPACK compression type

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  comp - Type of compression to be used in the approximate sparse factorization
      Possible values: NONE=0 HSS=1 BLR=2 HODLR=3 BLR_HODLR=4 ZFP_BLR_HODLR=5 LOSSLESS=6 LOSSY=7
      Default is NONE for -pc_type lu and BLR for -pc_type ilu

  Options Database:
.   -mat_strumpack_compression <NONE>  - Type of rank-structured compression in sparse LU factors (choose one of) NONE HSS BLR HODLR BLR_HODLR ZFP_BLR_HODLR LOSSLESS LOSSY (None)

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompression(Mat F,MatSTRUMPACKCompression comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(F,comp,2);
  PetscTryMethod(F,"MatSTRUMPACKSetCompression_C",(Mat,MatSTRUMPACKCompression),(F,comp));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompRelTol_STRUMPACK(Mat F,PetscReal rtol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression_rel_tol", STRUMPACK_set_compression_rel_tol(*S,rtol));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompRelTol - Set STRUMPACK relative tolerance for compression

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  rtol - relative compression tolerance

  Options Database:
.   -mat_strumpack_compression_rel_tol <1e-4>         - Relative compression tolerance, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompRelTol(Mat F,PetscReal rtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(F,rtol,2);
  PetscTryMethod(F,"MatSTRUMPACKSetCompRelTol_C",(Mat,PetscReal),(F,rtol));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompAbsTol_STRUMPACK(Mat F,PetscReal atol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression_abs_tol", STRUMPACK_set_compression_abs_tol(*S,atol));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompAbsTol - Set STRUMPACK absolute tolerance for compression

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  atol - absolute compression tolerance

  Options Database:
.   -mat_strumpack_compression_abs_tol <1e-10>         - Absolute compression tolerance, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompAbsTol(Mat F,PetscReal atol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(F,atol,2);
  PetscTryMethod(F,"MatSTRUMPACKSetCompAbsTol_C",(Mat,PetscReal),(F,atol));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompLeafSize_STRUMPACK(Mat F,PetscInt leaf_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression_leaf_size", STRUMPACK_set_compression_leaf_size(*S,leaf_size));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompLeafSize - Set STRUMPACK leaf size for HSS, BLR, HODLR, ...

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  leaf_size - Size of diagonal blocks in rank-structured approximation

  Options Database:
.   -mat_strumpack_compression_leaf_size    - Size of diagonal blocks in rank-structured approximation, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompLeafSize(Mat F,PetscInt leaf_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,leaf_size,2);
  PetscTryMethod(F,"MatSTRUMPACKSetCompLeafSize_C",(Mat,PetscInt),(F,leaf_size));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetGeometricNxyz_STRUMPACK(Mat F,PetscInt nx,PetscInt ny,PetscInt nz)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  if (nx < 1) {
    if (nx == PETSC_DECIDE || nx == PETSC_DEFAULT) nx = 1;
    else SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"nx < 1");
  }
  PetscStackCall("STRUMPACK_set_nx", STRUMPACK_set_nx(*S,nx));
  if (ny < 1) {
    if (ny == PETSC_DECIDE || ny == PETSC_DEFAULT) ny = 1;
    else SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"ny < 1");
  }
  PetscStackCall("STRUMPACK_set_ny", STRUMPACK_set_ny(*S,ny));
  if (nz < 1) {
    if (nz == PETSC_DECIDE || nz == PETSC_DEFAULT) nz = 1;
    else SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"nz < 1");
  }
  PetscStackCall("STRUMPACK_set_nz", STRUMPACK_set_nz(*S,nz));
  PetscFunctionReturn(0);
}
static PetscErrorCode MatSTRUMPACKSetGeometricComponents_STRUMPACK(Mat F,PetscInt nc)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_components", STRUMPACK_set_components(*S,nc));
  PetscFunctionReturn(0);
}
static PetscErrorCode MatSTRUMPACKSetGeometricWidth_STRUMPACK(Mat F,PetscInt w)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_separator_width", STRUMPACK_set_separator_width(*S,w));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetGeometricNxyz - Set STRUMPACK mesh x, y and z dimensions, for use with GEOMETRIC ordering.
   Logically Collective on Mat

   If the mesh is two (or one) dimensional one can use 1, PETSC_DECIDE or PETSC_DEFAULT
   for the missing z (and y) dimensions.

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  nx - x dimension of the mesh
-  ny - y dimension of the mesh
-  nz - z dimension of the mesh


   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetGeometricNxyz(Mat F,PetscInt nx,PetscInt ny,PetscInt nz)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,nx,2);
  PetscValidLogicalCollectiveInt(F,ny,2);
  PetscValidLogicalCollectiveInt(F,nz,2);
  PetscTryMethod(F,"MatSTRUMPACKSetGeometricNxyz_C",(Mat,PetscInt,PetscInt,PetscInt),(F,nx,ny,nz));
  PetscFunctionReturn(0);
}
/*@
  MatSTRUMPACKSetGeometricComponents - Set STRUMPACK number of degrees of freedom per mesh point, for use with GEOMETRIC ordering.
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  nc - Number of components/dof's per grid point

  Options Database:
.   -mat_strumpack_geometric_components <1>   - Number of components per mesh point, for geometric nested dissection ordering (None)

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetGeometricComponents(Mat F,PetscInt nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,nc,2);
  PetscTryMethod(F,"MatSTRUMPACKSetGeometricComponents_C",(Mat,PetscInt),(F,nc));
  PetscFunctionReturn(0);
}
/*@
  MatSTRUMPACKSetGeometricWidth - Set STRUMPACK width of the separator, for use with GEOMETRIC ordering.
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  w - width of the separator

  Options Database:
.   -mat_strumpack_geometric_width <1>        - Width of the separator of the mesh, for geometric nested dissection ordering (None)

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetGeometricWidth(Mat F,PetscInt w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,w,2);
  PetscTryMethod(F,"MatSTRUMPACKSetGeometricWidth_C",(Mat,PetscInt),(F,w));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompMinSepSize_STRUMPACK(Mat F,PetscInt min_sep_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression_min_sep_size", STRUMPACK_set_compression_min_sep_size(*S,min_sep_size));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompMinSepSize - Set STRUMPACK minimum separator size for low-rank approximation
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  min_sep_size - minimum dense matrix size for low-rank approximation

  Options Database:
.   -mat_strumpack_compression_min_sep_size <min_sep_size>    - Minimum size of dense sub-block for low-rank compression

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompMinSepSize(Mat F,PetscInt min_sep_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,min_sep_size,2);
  PetscTryMethod(F,"MatSTRUMPACKSetCompMinSepSize_C",(Mat,PetscInt),(F,min_sep_size));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompLossyPrecision_STRUMPACK(Mat F,PetscInt lossy_prec)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression_lossy_precision", STRUMPACK_set_compression_lossy_precision(*S,lossy_prec));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompLossyPrecision - Set STRUMPACK precision for lossy compression (requires ZFP support)
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  lossy_prec - Number of bitplanes to use in lossy compression

  Options Database:
.   -mat_strumpack_compression_lossy_precision <lossy_prec>    - Precision when using lossy compression [1-64], when using pctype ilu, compression LOSSY (None)

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompLossyPrecision(Mat F,PetscInt lossy_prec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,lossy_prec,2);
  PetscTryMethod(F,"MatSTRUMPACKSetCompLossyPrecision_C",(Mat,PetscInt),(F,lossy_prec));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetCompButterflyLevels_STRUMPACK(Mat F,PetscInt bfly_lvls)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_compression_butterfly_levels", STRUMPACK_set_compression_butterfly_levels(*S,bfly_lvls));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetCompButterflyLevels - Set STRUMPACK number of butterfly levels in HODLR compression (requires ButterflyPACK support)

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  bfly_lvls - Number of levels of butterfly compression in HODLR compression

  Options Database:
.   -mat_strumpack_compression_butterfly_levels <bfly_lvls>    - Number of levels in the hierarchically off-diagonal matrix for which to use butterfly, when using pctype ilu, (BLR_)HODLR compression (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetCompButterflyLevels(Mat F,PetscInt bfly_lvls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,bfly_lvls,2);
  PetscTryMethod(F,"MatSTRUMPACKSetButterflyLevels_C",(Mat,PetscInt),(F,bfly_lvls));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_STRUMPACK(Mat A,Vec b_mpi,Vec x)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  const PetscScalar      *bptr;
  PetscScalar            *xptr;

  PetscFunctionBegin;
  PetscCall(VecGetArray(x,&xptr));
  PetscCall(VecGetArrayRead(b_mpi,&bptr));

  PetscStackCall("STRUMPACK_solve",sp_err = STRUMPACK_solve(*S,(PetscScalar*)bptr,xptr,0));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: solve failed");
  }
  PetscCall(VecRestoreArray(x,&xptr));
  PetscCall(VecRestoreArrayRead(b_mpi,&bptr));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_STRUMPACK(Mat A,Mat B_mpi,Mat X)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  PetscBool              flg;
  PetscInt               m=A->rmap->n,nrhs;
  const PetscScalar      *bptr;
  PetscScalar            *xptr;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  PetscCall(MatGetSize(B_mpi,NULL,&nrhs));
  PetscCall(MatDenseGetArray(X,&xptr));
  PetscCall(MatDenseGetArrayRead(B_mpi,&bptr));

  PetscStackCall("STRUMPACK_solve",sp_err = STRUMPACK_matsolve(*S,nrhs,bptr,m,xptr,m,0));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: solve failed");
  }
  PetscCall(MatDenseRestoreArrayRead(B_mpi,&bptr));
  PetscCall(MatDenseRestoreArray(X,&xptr));

  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Info_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscFunctionBegin;
  /* check if matrix is strumpack type */
  if (A->ops->solve != MatSolve_STRUMPACK) PetscFunctionReturn(0);
  PetscCall(PetscViewerASCIIPrintf(viewer,"STRUMPACK sparse solver!\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_STRUMPACK(A,viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  Mat                    Aloc;
  const PetscScalar      *av;
  const PetscInt         *ai=NULL,*aj=NULL;
  PetscInt               M=A->rmap->N,m=A->rmap->n,dummy;
  PetscBool              ismpiaij,isseqaij,flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij));
  if (ismpiaij) {
    PetscCall(MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&Aloc));
  } else if (isseqaij) {
    PetscCall(PetscObjectReference((PetscObject)A));
    Aloc = A;
  } else SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);

  PetscCall(MatGetRowIJ(Aloc,0,PETSC_FALSE,PETSC_FALSE,&dummy,&ai,&aj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"GetRowIJ failed");
  PetscCall(MatSeqAIJGetArrayRead(Aloc,&av));

  if (ismpiaij) {
    MPI_Comm    comm;
    PetscMPIInt P,rank,p;
    PetscInt    *dist=NULL;
    PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
    PetscCallMPI(MPI_Comm_size(comm,&P));
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCall(PetscMalloc1(P+1,&dist));
    PetscCallMPI(MPI_Allgather(&m,1,MPIU_INT,dist+1,1,MPIU_INT,PETSC_COMM_WORLD));
    dist[0] = 0;
    for (p=0; p<P; p++) {
      dist[p+1] += dist[p];
    }
    PetscStackCall("STRUMPACK_set_distributed_csr_matrix",STRUMPACK_set_distributed_csr_matrix(*S,&m,ai,aj,av,dist,0));
    PetscCall(PetscFree(dist));
  } else if (isseqaij) {
    PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(*S,&M,ai,aj,av,0));
  } else SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);

  PetscCall(MatRestoreRowIJ(Aloc,0,PETSC_FALSE,PETSC_FALSE,&dummy,&ai,&aj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"RestoreRowIJ failed");
  PetscCall(MatSeqAIJRestoreArrayRead(Aloc,&av));
  PetscCall(MatDestroy(&Aloc));

  /* Reorder and Factor the matrix. */
  /* TODO figure out how to avoid reorder if the matrix values changed, but the pattern remains the same. */
  PetscStackCall("STRUMPACK_reorder",sp_err = STRUMPACK_reorder(*S));
  PetscStackCall("STRUMPACK_factor",sp_err = STRUMPACK_factor(*S));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: factorization failed");
  }
  F->assembled    = PETSC_TRUE;
  F->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  F->ops->solve           = MatSolve_STRUMPACK;
  F->ops->matsolve        = MatMatSolve_STRUMPACK;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_aij_strumpack(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSSTRUMPACK = "strumpack" - A solver package providing a direct sparse solver (PCLU)
  and a preconditioner (PCILU) using low-rank compression via the external package STRUMPACK.

  Consult the STRUMPACK manual for more info.

  Use
     ./configure --download-strumpack
  to have PETSc installed with STRUMPACK

  Use
    -pc_type lu -pc_factor_mat_solver_type strumpack
  to use this as an exact (direct) solver, use
    -pc_type ilu -pc_factor_mat_solver_type strumpack
  to enable low-rank compression (i.e, use as a preconditioner).

  Works with AIJ matrices

  Options Database Keys:
+ -mat_strumpack_verbose
. -mat_strumpack_compression                  - Type of rank-structured compression in sparse LU factors (choose one of) NONE HSS BLR HODLR BLR_HODLR ZFP_BLR_HODLR LOSSLESS LOSSY (None)
. -mat_strumpack_compression_rel_tol          - Relative compression tolerance, when using pctype ilu (None)
. -mat_strumpack_compression_abs_tol>         - Absolute compression tolerance, when using pctype ilu (None)
. -mat_strumpack_compression_min_sep_size     - Minimum size of separator for rank-structured compression, when using pctype ilu (None)
. -mat_strumpack_compression_leaf_size        - Size of diagonal blocks in rank-structured approximation, when using pctype ilu (None)
. -mat_strumpack_compression_lossy_precision  - Precision when using lossy compression [1-64], when using pctype ilu, compression LOSSY (None)
. -mat_strumpack_compression_butterfly_levels - Number of levels in the hierarchically off-diagonal matrix for which to use butterfly, when using pctype ilu, (BLR_)HODLR compression (None)
. -mat_strumpack_gpu                          - Enable GPU acceleration in numerical factorization (not supported for all compression types) (None)
. -mat_strumpack_colperm <TRUE>               - Permute matrix to make diagonal nonzeros (None)
. -mat_strumpack_reordering <METIS>           - Sparsity reducing matrix reordering (choose one of) NATURAL METIS PARMETIS SCOTCH PTSCOTCH RCM GEOMETRIC (None)
. -mat_strumpack_geometric_xyz <1,1,1>        - Mesh x,y,z dimensions, for use with GEOMETRIC ordering (None)
. -mat_strumpack_geometric_components <1>     - Number of components per mesh point, for geometric nested dissection ordering (None)
. -mat_strumpack_geometric_width <1>          - Width of the separator of the mesh, for geometric nested dissection ordering (None)
. -mat_strumpack_metis_nodeNDP                - Use METIS_NodeNDP instead of METIS_NodeND, for a more balanced tree (None)
- -mat_strumpack_iterative_solver <DIRECT>    - Select iterative solver from STRUMPACK (choose one of) AUTO DIRECT REFINE PREC_GMRES GMRES PREC_BICGSTAB BICGSTAB (None)

 Level: beginner

 HODLR, HODBF and BLR_HODBF compression require STRUMPACK to be configured with ButterflyPACK support.

 LOSSY and LOSSLESS compression require STRUMPACK to be configured with ZFP support.

.seealso: PCLU, PCILU, MATSOLVERSUPERLU_DIST, MATSOLVERMUMPS, PCFactorSetMatSolverType(), MatSolverType
M*/
static PetscErrorCode MatGetFactor_aij_strumpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                           B;
  PetscInt                      M=A->rmap->N,N=A->cmap->N;
  PetscBool                     verb,flg,set;
  PetscReal                     ctol;
  PetscInt                      min_sep_size,leaf_size,lossy_prec,bfly_lvls,nxyz[3],nrdims,nc,w;
  STRUMPACK_SparseSolver        *S;
  STRUMPACK_INTERFACE           iface;
  STRUMPACK_REORDERING_STRATEGY ndcurrent,ndvalue;
  STRUMPACK_KRYLOV_SOLVER       itcurrent,itsolver;
  STRUMPACK_COMPRESSION_TYPE    compcurrent,compvalue;
  const STRUMPACK_PRECISION     table[2][2][2] =
    {{{STRUMPACK_FLOATCOMPLEX_64, STRUMPACK_DOUBLECOMPLEX_64},
      {STRUMPACK_FLOAT_64,        STRUMPACK_DOUBLE_64}},
     {{STRUMPACK_FLOATCOMPLEX,    STRUMPACK_DOUBLECOMPLEX},
      {STRUMPACK_FLOAT,           STRUMPACK_DOUBLE}}};
  const STRUMPACK_PRECISION     prec = table[(sizeof(PetscInt)==8)?0:1][(PETSC_SCALAR==PETSC_COMPLEX)?0:1][(PETSC_REAL==PETSC_FLOAT)?0:1];
  const char *const             STRUMPACKNDTypes[] = {"NATURAL","METIS","PARMETIS","SCOTCH","PTSCOTCH","RCM","GEOMETRIC","STRUMPACKNDTypes","",0};
  const char *const             SolverTypes[] = {"AUTO","NONE","REFINE","PREC_GMRES","GMRES","PREC_BICGSTAB","BICGSTAB","SolverTypes","",0};
  const char *const             CompTypes[] = {"NONE","HSS","BLR","HODLR","BLR_HODLR","ZFP_BLR_HODLR","LOSSLESS","LOSSY","CompTypes","",0};

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,M,N));
  /* PetscCall(PetscStrallocpy("strumpack",&((PetscObject)B)->type_name)); */
  PetscCall(MatSetType(B,((PetscObject)A)->type_name));
  PetscCall(MatSetUp(B));
  PetscCall(MatSeqAIJSetPreallocation(B,0,NULL));
  PetscCall(MatMPIAIJSetPreallocation(B,0,NULL,0,NULL));
  B->trivialsymbolic = PETSC_TRUE;
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_STRUMPACK;
    B->ops->ilufactorsymbolic = MatLUFactorSymbolic_STRUMPACK;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  B->ops->view        = MatView_STRUMPACK;
  B->ops->destroy     = MatDestroy_STRUMPACK;
  B->ops->getdiagonal = MatGetDiagonal_STRUMPACK;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_aij_strumpack));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetReordering_C",MatSTRUMPACKSetReordering_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetColPerm_C",MatSTRUMPACKSetColPerm_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetGeometricNxyz_C",MatSTRUMPACKSetGeometricNxyz_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetGeometricComponents_C",MatSTRUMPACKSetGeometricComponents_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetGeometricWidth_C",MatSTRUMPACKSetGeometricWidth_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetGPU_C",MatSTRUMPACKSetGPU_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompression_C",MatSTRUMPACKSetCompression_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompRelTol_C",MatSTRUMPACKSetCompRelTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompAbsTol_C",MatSTRUMPACKSetCompAbsTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompLeafSize_C",MatSTRUMPACKSetCompLeafSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompMinSepSize_C",MatSTRUMPACKSetCompMinSepSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompLossyPrecision_C",MatSTRUMPACKSetCompLossyPrecision_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetCompButterflyLevels_C",MatSTRUMPACKSetCompButterflyLevels_STRUMPACK));
  B->factortype = ftype;

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSTRUMPACK,&B->solvertype));

  PetscCall(PetscNewLog(B,&S));
  B->spptr = S;

  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&flg));
  iface = flg ? STRUMPACK_MT : STRUMPACK_MPI_DIST;

  PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK Options","Mat");

  verb = PetscLogPrintInfo ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_strumpack_verbose","Print STRUMPACK information","None",verb,&verb,NULL));

  PetscStackCall("STRUMPACK_init",STRUMPACK_init(S,PetscObjectComm((PetscObject)A),prec,iface,0,NULL,verb));

  /* By default, no compression is done. Compression is enabled when the user enables it with        */
  /*  -mat_strumpack_compression with anything else than NONE, or when selecting ilu                 */
  /* preconditioning, in which case we default to STRUMPACK_BLR compression.                         */
  /* When compression is enabled, the STRUMPACK solver becomes an incomplete                         */
  /* (or approximate) LU factorization.                                                              */
  PetscStackCall("STRUMPACK_compression",compcurrent = STRUMPACK_compression(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_compression","Rank-structured compression type","None",CompTypes,(PetscEnum)compcurrent,(PetscEnum*)&compvalue,&set));
  if (set) {
    PetscStackCall("STRUMPACK_set_compression",STRUMPACK_set_compression(*S,compvalue));
  } else {
    if (ftype == MAT_FACTOR_ILU) {
      PetscStackCall("STRUMPACK_set_compression",STRUMPACK_set_compression(*S,STRUMPACK_BLR));
    }
  }

  PetscStackCall("STRUMPACK_compression_rel_tol",ctol = (PetscReal)STRUMPACK_compression_rel_tol(*S));
  PetscCall(PetscOptionsReal("-mat_strumpack_compression_rel_tol","Relative compression tolerance","None",ctol,&ctol,&set));
  if (set) PetscStackCall("STRUMPACK_set_compression_rel_tol",STRUMPACK_set_compression_rel_tol(*S,(double)ctol));

  PetscStackCall("STRUMPACK_compression_abs_tol",ctol = (PetscReal)STRUMPACK_compression_abs_tol(*S));
  PetscCall(PetscOptionsReal("-mat_strumpack_compression_abs_tol","Absolute compression tolerance","None",ctol,&ctol,&set));
  if (set) PetscStackCall("STRUMPACK_set_compression_abs_tol",STRUMPACK_set_compression_abs_tol(*S,(double)ctol));

  PetscStackCall("STRUMPACK_compression_min_sep_size",min_sep_size = (PetscInt)STRUMPACK_compression_min_sep_size(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_min_sep_size","Minimum size of separator for compression","None",min_sep_size,&min_sep_size,&set));
  if (set) PetscStackCall("STRUMPACK_set_compression_min_sep_size",STRUMPACK_set_compression_min_sep_size(*S,(int)min_sep_size));

  PetscStackCall("STRUMPACK_compression_leaf_size",leaf_size = (PetscInt)STRUMPACK_compression_leaf_size(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_leaf_size","Size of diagonal blocks in rank-structured approximation","None",leaf_size,&leaf_size,&set));
  if (set) PetscStackCall("STRUMPACK_set_compression_leaf_size",STRUMPACK_set_compression_leaf_size(*S,(int)leaf_size));

  PetscStackCall("STRUMPACK_compression_lossy_precision",lossy_prec = (PetscInt)STRUMPACK_compression_lossy_precision(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_lossy_precision","Number of bitplanes to use in lossy compression","None",lossy_prec,&lossy_prec,&set));
  if (set) PetscStackCall("STRUMPACK_set_compression_lossy_precision",STRUMPACK_set_compression_lossy_precision(*S,(int)lossy_prec));

  PetscStackCall("STRUMPACK_compression_butterfly_levels",bfly_lvls = (PetscInt)STRUMPACK_compression_butterfly_levels(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_butterfly_levels","Number of levels in the HODLR matrix for which to use butterfly compression","None",bfly_lvls,&bfly_lvls,&set));
  if (set) PetscStackCall("STRUMPACK_set_compression_butterfly_levels",STRUMPACK_set_compression_butterfly_levels(*S,(int)bfly_lvls));

  PetscStackCall("STRUMPACK_use_gpu",flg = (STRUMPACK_use_gpu(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_gpu","Enable GPU acceleration (not supported for all compression types)","None",flg,&flg,&set));
  if (set) {
    if (flg) {
      PetscStackCall("STRUMPACK_enable_gpu",STRUMPACK_enable_gpu(*S));
    } else {
      PetscStackCall("STRUMPACK_disable_gpu",STRUMPACK_disable_gpu(*S));
    }
  }

  PetscStackCall("STRUMPACK_matching",flg = (STRUMPACK_matching(*S) == STRUMPACK_MATCHING_NONE) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_colperm","Find a col perm to get nonzero diagonal","None",flg,&flg,&set));
  if (set) PetscStackCall("STRUMPACK_set_matching",STRUMPACK_set_matching(*S,flg ? STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING : STRUMPACK_MATCHING_NONE));

  PetscStackCall("STRUMPACK_reordering_method",ndcurrent = STRUMPACK_reordering_method(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_reordering","Sparsity reducing matrix reordering","None",STRUMPACKNDTypes,(PetscEnum)ndcurrent,(PetscEnum*)&ndvalue,&set));
  if (set) PetscStackCall("STRUMPACK_set_reordering_method",STRUMPACK_set_reordering_method(*S,ndvalue));

  /* geometric ordering, for a regular 1D/2D/3D mesh in the natural ordering, */
  /* with nc DOF's per gridpoint, and possibly a wider stencil                */
  nrdims = 3;
  nxyz[0] = nxyz[1] = nxyz[2] = 1;
  PetscCall(PetscOptionsIntArray("-mat_strumpack_geometric_xyz","Mesh sizes nx,ny,nz (Use 1 for default)","",nxyz,&nrdims,&set));
  if (set) {
    if (nrdims < 1 || nrdims > 3) {
      SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"'-mat_strumpack_geometrix_xyz' requires 1, 2, or 3 values.");
    }
    PetscStackCall("STRUMPACK_set_nx",STRUMPACK_set_nx(*S,(int)nxyz[0]));
    PetscStackCall("STRUMPACK_set_ny",STRUMPACK_set_ny(*S,(int)nxyz[1]));
    PetscStackCall("STRUMPACK_set_nz",STRUMPACK_set_nz(*S,(int)nxyz[2]));
  }
  PetscCall(PetscOptionsInt("-mat_strumpack_geometric_components","Number of components per mesh point, for geometric nested dissection ordering","None",1,&nc,&set));
  if (set) PetscStackCall("STRUMPACK_set_components",STRUMPACK_set_components(*S,(int)nc));
  PetscCall(PetscOptionsInt("-mat_strumpack_geometric_width","Width of the separator (for instance a 1D 3-point wide stencil needs a 1 point wide separator, a 1D 5-point stencil needs a 2 point wide separator), for geometric nested dissection ordering","None",1,&w,&set));
  if (set) PetscStackCall("STRUMPACK_set_separator_width",STRUMPACK_set_separator_width(*S,(int)w));

  PetscStackCall("STRUMPACK_use_METIS_NodeNDP",flg = (STRUMPACK_use_METIS_NodeNDP(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_metis_nodeNDP","Use METIS_NodeNDP instead of METIS_NodeND, for a more balanced tree","None",flg,&flg,&set));
  if (set) {
    if (flg) {
      PetscStackCall("STRUMPACK_enable_METIS_NodeNDP",STRUMPACK_enable_METIS_NodeNDP(*S));
    } else {
      PetscStackCall("STRUMPACK_disable_METIS_NodeNDP",STRUMPACK_disable_METIS_NodeNDP(*S));
    }
  }

  /* Disable the outer iterative solver from STRUMPACK.                                       */
  /* When STRUMPACK is used as a direct solver, it will by default do iterative refinement.   */
  /* When STRUMPACK is used as an approximate factorization preconditioner (by enabling       */
  /* low-rank compression), it will use it's own preconditioned GMRES. Here we can disable    */
  /* the outer iterative solver, as PETSc uses STRUMPACK from within a KSP.                   */
  PetscStackCall("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(*S, STRUMPACK_DIRECT));

  PetscStackCall("STRUMPACK_Krylov_solver",itcurrent = STRUMPACK_Krylov_solver(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_iterative_solver","Select iterative solver from STRUMPACK","None",SolverTypes,(PetscEnum)itcurrent,(PetscEnum*)&itsolver,&set));
  if (set) PetscStackCall("STRUMPACK_set_Krylov_solver",STRUMPACK_set_Krylov_solver(*S,itsolver));

  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_STRUMPACK(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATMPIAIJ,MAT_FACTOR_ILU,MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_ILU,MatGetFactor_aij_strumpack));
  PetscFunctionReturn(0);
}
