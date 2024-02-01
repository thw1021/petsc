#if !defined(PETSCDMBF_3D_ITERATE_H)
  #define PETSCDMBF_3D_ITERATE_H

  #include <petscdmbf.h> /*I "petscdmbf.h" I*/

PetscErrorCode DMBF_3D_IterateSetUpCells(DM, DM_BF_Cell *, const DM_BF_Shape *);
PetscErrorCode DMBF_3D_IterateSetUpP4estCells(DM, const DM_BF_Shape *);
PetscErrorCode DMBF_3D_IterateCopyP4estCells(DM, DM_BF_Cell *, const DM_BF_Shape *);
PetscErrorCode DMBF_3D_IterateSetCellData(DM, DM_BF_Cell *, size_t, size_t, size_t, const PetscInt *, PetscInt, const PetscInt *, PetscInt, Vec *, Vec *);
PetscErrorCode DMBF_3D_IterateSetCellFields(DM, DM_BF_Cell *, size_t, size_t, size_t, const PetscInt *, PetscInt, const PetscInt *, PetscInt, Vec *, Vec *, PetscInt, PetscInt *, PetscInt, PetscInt *);
PetscErrorCode DMBF_3D_IterateGetCellData(DM, DM_BF_Cell *, size_t, size_t, size_t, const PetscInt *, PetscInt, const PetscInt *, PetscInt, Vec *, Vec *);
PetscErrorCode DMBF_3D_IterateGetCellFields(DM, DM_BF_Cell *, size_t, size_t, size_t, const PetscInt *, PetscInt, const PetscInt *, PetscInt, Vec *, Vec *, PetscInt, PetscInt *, PetscInt, PetscInt *);
PetscErrorCode DMBF_3D_IterateGhostExchange(DM, DM_BF_Cell *, size_t);

PetscErrorCode DMBF_3D_IterateOverCellsVectors(DM, DM_BF_Cell *, size_t, PetscErrorCode (*)(DM, DM_BF_Cell *, void *), void *, Vec *, PetscInt, Vec *, PetscInt);
PetscErrorCode DMBF_3D_IterateOverFaces(DM, DM_BF_Cell *, size_t, PetscErrorCode (*)(DM, DM_BF_Face *, void *), void *);
PetscErrorCode DMBF_3D_IterateFVMatAssembly(DM, DM_BF_Cell *, size_t, Mat, PetscErrorCode (*)(DM, DM_BF_Face *, PetscReal *, void *), void *);

#endif /* defined(PETSCDMBF_3D_ITERATE_H) */
