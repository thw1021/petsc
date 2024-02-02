#include <petscdmbf.h> /*I "petscdmbf.h" I*/

#if defined(PETSC_HAVE_P4EST)

typedef struct _p_DM_BF_AmrCtx {
  int minLevel;
  int maxLevel;
} DM_BF_AmrCtx;

/***************************************
 * UNIFORM COARSENING/REFINEMENT
 **************************************/

static int _p_coarsen_uniformly(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  DM_BF_AmrCtx *amrCtx = p4est->user_pointer;

  //SC_CHECK_ABORT (p4est_quadrant_is_familypv (quadrants), "Coarsen invocation");
  return (0 <= amrCtx->minLevel && amrCtx->minLevel < quadrants[0]->level);
}

static int _p_refine_uniformly(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  DM_BF_AmrCtx *amrCtx = p4est->user_pointer;

  return (0 <= amrCtx->maxLevel && quadrant->level < amrCtx->maxLevel);
}

  #if !defined(DMBF_XD_AmrCoarsenUniformly)
static
  #endif
  PetscErrorCode
  DMBF_XD_AmrCoarsenUniformly(p4est_t *p4est, PetscInt minLevel)
{
  void        *user_pointer = p4est->user_pointer;
  DM_BF_AmrCtx amrCtx;

  PetscFunctionBegin;
  /* set AMR context */
  amrCtx.minLevel     = (int)minLevel;
  amrCtx.maxLevel     = -1;
  p4est->user_pointer = (void *)&amrCtx;
  /* run AMR */
  PetscCallP4est(p4est_coarsen, (p4est, 0 /*!recursively*/, _p_coarsen_uniformly, NULL /*init_fn*/));
  p4est->user_pointer = user_pointer;
  /* balance and partition */
  PetscCallP4est(p4est_balance, (p4est, P4EST_CONNECT_FULL, NULL /*init_fn*/));
  PetscCallP4est(p4est_partition_ext, (p4est, 1 /*partition_for_coarsening*/, NULL /*weight_fn*/));
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_AmrRefineUniformly)
static
  #endif
  PetscErrorCode
  DMBF_XD_AmrRefineUniformly(p4est_t *p4est, PetscInt maxLevel)
{
  void        *user_pointer = p4est->user_pointer;
  DM_BF_AmrCtx amrCtx;

  PetscFunctionBegin;
  /* set AMR context */
  amrCtx.minLevel     = -1;
  amrCtx.maxLevel     = (int)maxLevel;
  p4est->user_pointer = (void *)&amrCtx;
  /* run AMR */
  PetscCallP4est(p4est_refine, (p4est, 0 /*!recursively*/, _p_refine_uniformly, NULL /*init_fn*/));
  p4est->user_pointer = user_pointer;
  /* balance and partition */
  //PetscCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL/*init_fn*/));
  PetscCallP4est(p4est_partition_ext, (p4est, 1 /*partition_for_coarsening*/, NULL /*weight_fn*/));
  PetscFunctionReturn(0);
}

/***************************************
 * ADAPTIVE COARSENING/REFINEMENT
 **************************************/

  #if 0
static int
_p_flag_is_valid (DMAdaptFlag flag)
{
  switch (flag) {
    case DM_ADAPT_KEEP:
    case DM_ADAPT_COARSEN:
    case DM_ADAPT_REFINE:
      return 1;
    default: /* unknown flag */
      return 0;
  }
}
  #endif

static int _p_coarsen_via_flag(p4est_t *p4est, p4est_topidx_t tree, p4est_quadrant_t *quadrants[])
{
  DM_BF_AmrCtx *amrCtx = p4est->user_pointer;
  int           k;

  //SC_CHECK_ABORT (p4est_quadrant_is_familypv (quadrants), "Coarsen invocation");
  for (k = 0; k < P4EST_CHILDREN; k++) {
    DM_BF_Cell *cell = quadrants[k]->p.user_data;
    if (!cell) { return 0; }
    /* if at least one child is not flagged for coarsening */
    if (DM_ADAPT_COARSEN != cell->adaptFlag) { return 0; }
  }
  /* if all of the children are flagged for coarsening */
  return (0 <= amrCtx->minLevel && amrCtx->minLevel < quadrants[0]->level);
}

static int _p_refine_via_flag(p4est_t *p4est, p4est_topidx_t tree, p4est_quadrant_t *quadrant)
{
  DM_BF_AmrCtx *amrCtx = p4est->user_pointer;
  DM_BF_Cell   *cell   = quadrant->p.user_data;
  if (!cell) { return 0; }
  /* if this quadrant is flagged for refinement */
  return (DM_ADAPT_REFINE == cell->adaptFlag) && (0 <= amrCtx->maxLevel && quadrant->level < amrCtx->maxLevel);
}

  #if !defined(DMBF_XD_AmrAdapt)
static
  #endif
  PetscErrorCode
  DMBF_XD_AmrAdapt(p4est_t *p4est, PetscInt minLevel, PetscInt maxLevel)
{
  void        *user_pointer = p4est->user_pointer;
  DM_BF_AmrCtx amrCtx;

  PetscFunctionBegin;
  /* set AMR context */
  amrCtx.minLevel     = (int)minLevel;
  amrCtx.maxLevel     = (int)maxLevel;
  p4est->user_pointer = (void *)&amrCtx;
  /* run AMR */
  PetscCallP4est(p4est_coarsen, (p4est, 0 /*!recursively*/, _p_coarsen_via_flag, NULL /*init_fn*/));
  PetscCallP4est(p4est_refine, (p4est, 0 /*!recursively*/, _p_refine_via_flag, NULL /*init_fn*/));
  p4est->user_pointer = user_pointer;
  /* balance */
  PetscCallP4est(p4est_balance, (p4est, P4EST_CONNECT_FULL, NULL /*init_fn*/));
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_AmrAdaptData)
static
  #endif
  PetscErrorCode
  DMBF_XD_AmrAdaptData(p4est_t *orig_p4est, p4est_t *adap_p4est, DM dm, DM_BF_AmrOps *amrOps)
{
  const p4est_locidx_t orig_n_quads = orig_p4est->local_num_quadrants;
  const p4est_locidx_t adap_n_quads = adap_p4est->local_num_quadrants;
  p4est_locidx_t       orig_quadid, adap_quadid;
  int8_t               orig_level, adap_level;
  p4est_quadrant_t    *orig_quad, *adap_quad;
  int                  i;
  const size_t         cellSize = orig_p4est->data_size;
  DM_BF_Cell          *orig_cell[P4EST_CHILDREN], *adap_cell[P4EST_CHILDREN];
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* check input */
  PetscCheck(orig_p4est->data_size == adap_p4est->data_size, orig_p4est->mpicomm, PETSC_ERR_ARG_SIZ, "p4est data size mismatch: original %d, adapted %d", (int)orig_p4est->data_size, (int)adap_p4est->data_size);
  PetscCheck(amrOps->projectToCoarse, orig_p4est->mpicomm, PETSC_ERR_ARG_NULL, "Project function to coarse is not given");
  PetscCheck(amrOps->projectToFine, orig_p4est->mpicomm, PETSC_ERR_ARG_NULL, "Project function to fine is not given");
  /* loop over all p4est quadrants */
  orig_quadid = 0;
  adap_quadid = 0;
  while (orig_quadid < orig_n_quads) {
    PetscCheck(adap_quadid < adap_n_quads, orig_p4est->mpicomm, PETSC_ERR_PLIB, "Quadrant id %d is larger than the number of quadrants %d", (int)adap_quadid, (int)adap_n_quads);
    PetscCallP4estReturn(orig_quad, p4est_find_quadrant_cumulative, (orig_p4est, orig_quadid, NULL, NULL));
    PetscCallP4estReturn(adap_quad, p4est_find_quadrant_cumulative, (adap_p4est, adap_quadid, NULL, NULL));
    orig_level = orig_quad->level;
    adap_level = adap_quad->level;
    if (adap_level < orig_level) { /* if this element was coarsened */
      for (i = 0; i < P4EST_CHILDREN; i++) {
        PetscCallP4estReturn(orig_quad, p4est_find_quadrant_cumulative, (orig_p4est, orig_quadid + i, NULL, NULL));
        orig_cell[i] = (DM_BF_Cell *)orig_quad->p.user_data;
      }
      adap_cell[0] = (DM_BF_Cell *)adap_quad->p.user_data;
      ierr         = amrOps->projectToCoarse(dm, orig_cell, P4EST_CHILDREN, adap_cell, 1, amrOps->projectToCoarseCtx);
      CHKERRQ(ierr);
      /* skip the next 2^dim fine elements of the original mesh */
      orig_quadid += P4EST_CHILDREN;
      /* go to the next element of the adapted mesh */
      adap_quadid += 1;
    } else if (orig_level < adap_level) { /* if this element was refined */
      orig_cell[0] = (DM_BF_Cell *)orig_quad->p.user_data;
      for (i = 0; i < P4EST_CHILDREN; i++) {
        PetscCallP4estReturn(adap_quad, p4est_find_quadrant_cumulative, (adap_p4est, adap_quadid + i, NULL, NULL));
        adap_cell[i] = (DM_BF_Cell *)adap_quad->p.user_data;
      }
      ierr = amrOps->projectToFine(dm, orig_cell, 1, adap_cell, P4EST_CHILDREN, amrOps->projectToFineCtx);
      CHKERRQ(ierr);
      /* go to the next element of the original mesh */
      orig_quadid += 1;
      /* skip the next 2^dim fine elements of the adapted mesh */
      adap_quadid += P4EST_CHILDREN;
    } else { /* otherwise this element has not changed */
      orig_cell[0] = (DM_BF_Cell *)orig_quad->p.user_data;
      adap_cell[0] = (DM_BF_Cell *)adap_quad->p.user_data;
      ierr         = PetscMemcpy(adap_cell[0], orig_cell[0], cellSize);
      CHKERRQ(ierr);
      /* go to next element */
      orig_quadid += 1;
      adap_quadid += 1;
    }
  }
  /* check final quadrant id's */
  PetscCheck(orig_quadid == orig_n_quads, orig_p4est->mpicomm, PETSC_ERR_PLIB, "Original quadrant id %d did not reach number of quadrants %d", (int)orig_quadid, (int)orig_n_quads);
  PetscCheck(adap_quadid == adap_n_quads, orig_p4est->mpicomm, PETSC_ERR_PLIB, "Adapted quadrant id %d did not reach number of quadrants %d", (int)adap_quadid, (int)adap_n_quads);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_AmrPartition)
static
  #endif
  PetscErrorCode
  DMBF_XD_AmrPartition(p4est_t *p4est)
{
  PetscFunctionBegin;
  PetscCallP4est(p4est_partition_ext, (p4est, 1 /*partition_for_coarsening*/, NULL /*weight_fn*/));
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
