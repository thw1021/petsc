/* Internal and DMStag-specific functions related to multigrid */
#include <petsc/private/dmstagimpl.h>

/*@C
  DMStagRestrictSimple - restricts data from a fine to a coarse DMStag, in the simplest way

  Values on coarse cells are averages of all fine cells that they cover.
  Thus, values on vertices are injected, values on edges are averages
  of the underlying two fine edges, and and values on elements in
  d dimensions are averages of 2^d underlying elements.

  Input Arguments:
  + dmf - fine DM
  . xf - data on fine DM
  - dmc - coarse DM

  Output Arguments:
  - xc - data on coarse DM

  Level: advanced

  .seealso: DMRestrict(), DMCoarsen(), DMSTAG, DMCreateInjection()
@*/
PetscErrorCode DMStagRestrictSimple(DM dmf,Vec xf,DM dmc,Vec xc)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBegin;
  ierr = DMGetDimension(dmf,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      ierr = DMStagRestrictSimple_1d(dmf,xf,dmc,xc);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMStagRestrictSimple_2d(dmf,xf,dmc,xc);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
      break;
  }
  PetscFunctionReturn(0);
}

/* Code duplication note: the next two functions are nearly identical, save the inclusion of the element terms */
PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation1d_1_0_Private(DM dmc,DM dmf,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       exf,startexf,nexf,nextraxf,startexc;

  PetscFunctionBegin;

  /* In 1D, each fine point can receive data from at most 2 coarse points, at most one of which could be off-process */
  ierr = MatSeqAIJSetPreallocation(A,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,2,NULL,1,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,NULL,NULL,&nexf,NULL,NULL,&nextraxf,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
    PetscInt exc,exf_local;
    exf_local = exf-startexf;
    exc = startexc + exf_local/2;
    /* "even" vertices are just injected, odd vertices averaged */
    if (exf_local % 2 == 0) {
      DMStagStencil     rowf,colc;
      PetscInt          ir,ic;
      const PetscScalar one = 1.0;
      rowf.i = exf; rowf.c = 0; rowf.loc = DMSTAG_LEFT;
      colc.i = exc; colc.c = 0; colc.loc = DMSTAG_LEFT;
      ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmc,1,&colc,&ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&one,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      DMStagStencil     rowf,colc[2];
      PetscInt          ir,ic[2];
      const PetscScalar halves[2] = {0.5,0.5};
      rowf.i    = exf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
      colc[0].i = exc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
      colc[1].i = exc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
      ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmc,2, colc, ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&ir,2,ic,halves,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation1d_1_1_Private(DM dmc,DM dmf,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       exf,startexf,nexf,nextraxf,startexc;

  PetscFunctionBegin;

  /* In 1D, each fine point can receive data from at most 2 coarse points, at most one of which could be off-process */
  ierr = MatSeqAIJSetPreallocation(A,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,2,NULL,1,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,NULL,NULL,&nexf,NULL,NULL,&nextraxf,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
    PetscInt exc,exf_local;
    exf_local = exf-startexf;
    exc = startexc + exf_local/2;
    /* Elements (excluding "extra" dummies) */
    if (exf < startexf+nexf) {
      DMStagStencil     rowf,colc;
      PetscInt          ir,ic;
      const PetscScalar one = 1.0;
      rowf.i = exf;  rowf.c = 0; rowf.loc = DMSTAG_ELEMENT; /* Note that this assumes only 1 dof */
      colc.i = exc;  colc.c = 0; colc.loc = DMSTAG_ELEMENT;
      ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmc,1,&colc,&ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* "even" vertices are just injected, odd vertices averaged */
    if (exf_local % 2 == 0) {
      DMStagStencil     rowf,colc;
      PetscInt          ir,ic;
      const PetscScalar one = 1.0;
      rowf.i = exf; rowf.c = 0; rowf.loc = DMSTAG_LEFT;
      colc.i = exc; colc.c = 0; colc.loc = DMSTAG_LEFT;
      ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmc,1,&colc,&ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&one,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      DMStagStencil     rowf,colc[2];
      PetscInt          ir,ic[2];
      const PetscScalar halves[2] = {0.5,0.5};
      rowf.i    = exf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
      colc[0].i = exc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
      colc[1].i = exc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
      ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmc,2, colc, ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&ir,2,ic,halves,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Code duplication note: the next two functions are almost exactly the same, save the inclusion of the element terms */
PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation2d_0_1_0_Private(DM dmc,DM dmf,Mat A)
{
  PetscErrorCode ierr;
  PetscInt exf,eyf,startexf,starteyf,nexf,neyf,nextraxf,nextrayf,startexc,starteyc,Nexf,Neyf;

  /* In 2D, each fine point can receive data from at most 4 coarse points, at most 3 of which could be off-process */
  ierr = MatSeqAIJSetPreallocation(A,4,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,4,NULL,3,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,&starteyf,NULL,&nexf,&neyf,NULL,&nextraxf,&nextrayf,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,&starteyc,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,NULL);CHKERRQ(ierr);
  for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
    PetscInt eyc,eyf_local;
    eyf_local = eyf-starteyf;
    eyc = starteyc + eyf_local/2;
    for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
      PetscInt exc,exf_local;
      exf_local = exf-startexf;
      exc = startexc + exf_local/2;
      /* Left edges (excluding top "extra" dummy row) */
      if (eyf < starteyf+neyf) {
        DMStagStencil rowf,colc[4];
        PetscInt      ir,ic[4],nweight;
        PetscScalar   weight[4];
        rowf.i    = exf; rowf.j    = eyf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
        colc[0].i = exc; colc[0].j = eyc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
        if (exf_local % 2 == 0) {
          if (eyf == Neyf-1 || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 1; weight[0] = 1.0;
          } else {
            nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
            if (eyf_local % 2 == 0) {
              colc[1].i = exc; colc[1].j = eyc-1; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
            } else {
              colc[1].i = exc; colc[1].j = eyc+1; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
            }
          }
        } else {
          colc[1].i = exc; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
          if (eyf == Neyf-1 || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 2; weight[0] = 0.5; weight[1] = 0.5;
          } else {
            nweight = 4; weight[0] = 0.375; weight[1] = 0.375; weight[2] = 0.125; weight[3] = 0.125;
            if (eyf_local % 2 == 0) {
              colc[2].i = exc; colc[2].j = eyc-1; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
              colc[3].i = exc; colc[3].j = eyc-1; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
            } else {
              colc[2].i = exc; colc[2].j = eyc+1; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
              colc[3].i = exc; colc[3].j = eyc+1; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
            }
          }
        }
        ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmc,nweight,colc,ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Down edges (excluding right "extra" dummy col) */
      if (exf < startexf+nexf) {
        DMStagStencil rowf,colc[4];
        PetscInt      ir,ic[4],nweight;
        PetscScalar   weight[4];
        rowf.i    = exf; rowf.j    = eyf; rowf.c    = 0; rowf.loc    = DMSTAG_DOWN;
        colc[0].i = exc; colc[0].j = eyc; colc[0].c = 0; colc[0].loc = DMSTAG_DOWN;
        if (eyf_local % 2 == 0) {
          if (exf == Nexf-1 || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 1; weight[0] = 1.0;
          } else {
            nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
            if (exf_local % 2 == 0) {
              colc[1].i = exc-1; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
            } else {
              colc[1].i = exc+1; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
            }
          }
        } else {
          colc[1].i = exc; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_UP;
          if (exf == Nexf-1 || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 2; weight[0] = 0.5; weight[1] = 0.5;
          } else {
            nweight = 4; weight[0] = 0.375; weight[1] = 0.375; weight[2] = 0.125; weight[3] = 0.125;
            if (exf_local % 2 == 0) {
              colc[2].i = exc-1; colc[2].j = eyc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
              colc[3].i = exc-1; colc[3].j = eyc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
            } else {
              colc[2].i = exc+1; colc[2].j = eyc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
              colc[3].i = exc+1; colc[3].j = eyc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
            }
          }
        }
        ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmc,nweight,colc,ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation2d_0_1_1_Private(DM dmc,DM dmf,Mat A)
{
  PetscErrorCode ierr;
  PetscInt exf,eyf,startexf,starteyf,nexf,neyf,nextraxf,nextrayf,startexc,starteyc,Nexf,Neyf;

  /* In 2D, each fine point can receive data from at most 4 coarse points, at most 3 of which could be off-process */
  ierr = MatSeqAIJSetPreallocation(A,4,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,4,NULL,3,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,&starteyf,NULL,&nexf,&neyf,NULL,&nextraxf,&nextrayf,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,&starteyc,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,NULL);CHKERRQ(ierr);
  for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
    PetscInt eyc,eyf_local;
    eyf_local = eyf-starteyf;
    eyc = starteyc + eyf_local/2;
    for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
      PetscInt exc,exf_local;
      exf_local = exf-startexf;
      exc = startexc + exf_local/2;
      /* Elements (excluding "extra" dummy) */
      if (exf < startexf+nexf && eyf < starteyf+neyf) {
        DMStagStencil     rowf,colc;
        PetscInt          ir,ic;
        const PetscScalar one = 1.0;
        rowf.i = exf; rowf.j = eyf; rowf.c = 0; rowf.loc = DMSTAG_ELEMENT;
        colc.i = exc; colc.j = eyc; colc.c = 0; colc.loc = DMSTAG_ELEMENT;
        ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmc,1,&colc,&ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&one,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Left edges (excluding top "extra" dummy row) */
      if (eyf < starteyf+neyf) {
        DMStagStencil rowf,colc[4];
        PetscInt      ir,ic[4],nweight;
        PetscScalar   weight[4];
        rowf.i    = exf; rowf.j    = eyf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
        colc[0].i = exc; colc[0].j = eyc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
        if (exf_local % 2 == 0) {
          if (eyf == Neyf-1 || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 1; weight[0] = 1.0;
          } else {
            nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
            if (eyf_local % 2 == 0) {
              colc[1].i = exc; colc[1].j = eyc-1; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
            } else {
              colc[1].i = exc; colc[1].j = eyc+1; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
            }
          }
        } else {
          colc[1].i = exc; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
          if (eyf == Neyf-1 || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 2; weight[0] = 0.5; weight[1] = 0.5;
          } else {
            nweight = 4; weight[0] = 0.375; weight[1] = 0.375; weight[2] = 0.125; weight[3] = 0.125;
            if (eyf_local % 2 == 0) {
              colc[2].i = exc; colc[2].j = eyc-1; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
              colc[3].i = exc; colc[3].j = eyc-1; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
            } else {
              colc[2].i = exc; colc[2].j = eyc+1; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
              colc[3].i = exc; colc[3].j = eyc+1; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
            }
          }
        }
        ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmc,nweight,colc,ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Down edges (excluding right "extra" dummy col) */
      if (exf < startexf+nexf) {
        DMStagStencil rowf,colc[4];
        PetscInt      ir,ic[4],nweight;
        PetscScalar   weight[4];
        rowf.i    = exf; rowf.j    = eyf; rowf.c    = 0; rowf.loc    = DMSTAG_DOWN;
        colc[0].i = exc; colc[0].j = eyc; colc[0].c = 0; colc[0].loc = DMSTAG_DOWN;
        if (eyf_local % 2 == 0) {
          if (exf == Nexf-1 || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 1; weight[0] = 1.0;
          } else {
            nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
            if (exf_local % 2 == 0) {
              colc[1].i = exc-1; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
            } else {
              colc[1].i = exc+1; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
            }
          }
        } else {
          colc[1].i = exc; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_UP;
          if (exf == Nexf-1 || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 2; weight[0] = 0.5; weight[1] = 0.5;
          } else {
            nweight = 4; weight[0] = 0.375; weight[1] = 0.375; weight[2] = 0.125; weight[3] = 0.125;
            if (exf_local % 2 == 0) {
              colc[2].i = exc-1; colc[2].j = eyc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
              colc[3].i = exc-1; colc[3].j = eyc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
            } else {
              colc[2].i = exc+1; colc[2].j = eyc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
              colc[3].i = exc+1; colc[3].j = eyc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
            }
          }
        }
        ierr = DMStagStencilToIndexLocal(dmf,1,&rowf,&ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmc,nweight,colc,ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction1d_1_0_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt exf,startexf,nexf,nextraxf,startexc,Nexf;

  PetscErrorCode ierr;

  /* In 1D, each coarse point can receive from up to 3 fine points, one of which may be off-rank */
  ierr = MatSeqAIJSetPreallocation(A,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,NULL,1,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,NULL,NULL,&nexf,NULL,NULL,&nextraxf,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmf,&Nexf,NULL,NULL);CHKERRQ(ierr);
  for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
    PetscInt exc,exf_local;
    exf_local = exf-startexf;
    exc = startexc + exf_local/2;
    /* "even" vertices contribute to the overlying coarse vertex, odd vertices to the two adjacent */
    if (exf_local % 2 == 0) {
      DMStagStencil colf,rowc;
      PetscInt      ir,ic;
      PetscScalar   weight = 0.5;
      colf.i = exf; colf.c = 0; colf.loc = DMSTAG_LEFT;
      rowc.i = exc; rowc.c = 0; rowc.loc = DMSTAG_LEFT;
      ierr = DMStagStencilToIndexLocal(dmc,1,&rowc,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
      weight = (exf == Nexf || exf == 0) ? 0.75 : 0.5; /* Assume a Neuman-type condition */
      ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      DMStagStencil     colf,rowc[2];
      PetscInt          ic,ir[2];
      const PetscScalar quarters[2] = {0.25,0.25};
      colf.i    = exf;  colf.c    = 0; colf.loc    = DMSTAG_LEFT;
      rowc[0].i = exc;  rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
      rowc[1].i = exc;  rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
      ierr = DMStagStencilToIndexLocal(dmc,2, rowc, ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,2,ir,1,&ic,quarters,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction1d_1_1_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt exf,startexf,nexf,nextraxf,startexc,Nexf;

  PetscErrorCode ierr;

  /* In 1D, each coarse point can receive from up to 3 fine points, one of which may be off-rank */
  ierr = MatSeqAIJSetPreallocation(A,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,NULL,1,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,NULL,NULL,&nexf,NULL,NULL,&nextraxf,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmf,&Nexf,NULL,NULL);CHKERRQ(ierr);
  for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
    PetscInt exc,exf_local;
    exf_local = exf-startexf;
    exc = startexc + exf_local/2;
    if (exf < startexf+nexf) {
      DMStagStencil     rowc,colf;
      PetscInt          ir,ic;
      const PetscScalar cellScale = 0.5;
      rowc.i = exc; rowc.c = 0; rowc.loc = DMSTAG_ELEMENT;
      colf.i = exf; colf.c = 0; colf.loc = DMSTAG_ELEMENT;
      ierr = DMStagStencilToIndexLocal(dmc,1,&rowc,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&cellScale,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* "even" vertices contribute to the overlying coarse vertex, odd vertices to the two adjacent */
    if (exf_local % 2 == 0) {
      DMStagStencil colf,rowc;
      PetscInt      ir,ic;
      PetscScalar   weight;
      colf.i = exf; colf.c = 0; colf.loc = DMSTAG_LEFT;
      rowc.i = exc; rowc.c = 0; rowc.loc = DMSTAG_LEFT;
      ierr = DMStagStencilToIndexLocal(dmc,1,&rowc,&ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
      weight = (exf == Nexf || exf == 0) ? 0.75 : 0.5; /* Assume a Neuman-type condition */
      ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      DMStagStencil     colf,rowc[2];
      PetscInt          ic,ir[2];
      const PetscScalar quarters[2] = {0.25,0.25};
      colf.i    = exf;  colf.c    = 0; colf.loc    = DMSTAG_LEFT;
      rowc[0].i = exc;  rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
      rowc[1].i = exc;  rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
      ierr = DMStagStencilToIndexLocal(dmc,2, rowc, ir);CHKERRQ(ierr);
      ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(A,2,ir,1,&ic,quarters,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction2d_0_1_0_Private(DM dmc,DM dmf,Mat A)
{
  PetscErrorCode ierr;
  PetscInt exf,eyf,startexf,starteyf,nexf,neyf,nextraxf,nextrayf,startexc,starteyc,Nexf,Neyf;

  PetscFunctionBegin;

  /* In 2D, each coarse point can receive from up to 6 fine points,
     up to 2 of which may be off rank */
  ierr = MatSeqAIJSetPreallocation(A,6,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,6,NULL,2,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,&starteyf,NULL,&nexf,&neyf,NULL,&nextraxf,&nextrayf,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,&starteyc,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,NULL);CHKERRQ(ierr);
  for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
    PetscInt eyc,eyf_local;
    eyf_local = eyf-starteyf;
    eyc = starteyc + eyf_local/2;
    for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
      PetscInt exc,exf_local;
      exf_local = exf-startexf;
      exc = startexc + exf_local/2;
      /* Left edges (excluding top "extra" dummy row) */
      if (eyf < starteyf+neyf) {
        DMStagStencil rowc[2],colf;
        PetscInt      ir[2],ic,nweight;
        PetscScalar   weight[2];
        colf.i    = exf; colf.j    = eyf; colf.c = 0; colf.loc       = DMSTAG_LEFT;
        rowc[0].i = exc; rowc[0].j = eyc; rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
        if (exf_local % 2 == 0) {
          nweight = 1;
          if (exf == Nexf || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            weight[0] = 0.375;
          } else {
            weight[0] = 0.25;
          }
        } else {
          nweight = 2;
          rowc[1].i = exc; rowc[1].j = eyc; rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
          weight[0] = 0.125; weight[1] = 0.125;
        }
        ierr = DMStagStencilToIndexLocal(dmc,nweight,rowc,ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Down edges (excluding right "extra" dummy col) */
      if (exf < startexf+nexf) {
        DMStagStencil rowc[2],colf;
        PetscInt      ir[2],ic,nweight;
        PetscScalar   weight[2];
        colf.i    = exf; colf.j    = eyf; colf.c = 0; colf.loc       = DMSTAG_DOWN;
        rowc[0].i = exc; rowc[0].j = eyc; rowc[0].c = 0; rowc[0].loc = DMSTAG_DOWN;
        if (eyf_local % 2 == 0) {
          nweight = 1;
          if (eyf == Neyf || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            weight[0] = 0.375;
          } else {
            weight[0] = 0.25;
          }
        } else {
          nweight = 2;
          rowc[1].i = exc; rowc[1].j = eyc; rowc[1].c = 0; rowc[1].loc = DMSTAG_UP;
          weight[0] = 0.125; weight[1] = 0.125;
        }
        ierr = DMStagStencilToIndexLocal(dmc,nweight,rowc,ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction2d_0_1_1_Private(DM dmc,DM dmf,Mat A)
{
  PetscErrorCode ierr;
  PetscInt exf,eyf,startexf,starteyf,nexf,neyf,nextraxf,nextrayf,startexc,starteyc,Nexf,Neyf;

  PetscFunctionBegin;

  /* In 2D, each coarse point can receive from up to 6 fine points,
     up to 2 of which may be off rank */
  ierr = MatSeqAIJSetPreallocation(A,6,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,6,NULL,2,NULL);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmf,&startexf,&starteyf,NULL,&nexf,&neyf,NULL,&nextraxf,&nextrayf,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmc,&startexc,&starteyc,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,NULL);CHKERRQ(ierr);
  for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
    PetscInt eyc,eyf_local;
    eyf_local = eyf-starteyf;
    eyc = starteyc + eyf_local/2;
    for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
      PetscInt exc,exf_local;
      exf_local = exf-startexf;
      exc = startexc + exf_local/2;
      /* Elements (excluding "extra" dummies) */
      if (exf < startexf+nexf && eyf < starteyf+neyf) {
        DMStagStencil     rowc,colf;
        PetscInt          ir,ic;
        const PetscScalar cellScale = 0.25;
        rowc.i = exc; rowc.j = eyc; rowc.c = 0; rowc.loc = DMSTAG_ELEMENT;
        colf.i = exf; colf.j = eyf; colf.c = 0; colf.loc = DMSTAG_ELEMENT;
        ierr = DMStagStencilToIndexLocal(dmc,1,&rowc,&ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,1,&ir,1,&ic,&cellScale,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Left edges (excluding top "extra" dummy row) */
      if (eyf < starteyf+neyf) {
        DMStagStencil rowc[2],colf;
        PetscInt      ir[2],ic,nweight;
        PetscScalar   weight[2];
        colf.i    = exf; colf.j    = eyf; colf.c = 0; colf.loc       = DMSTAG_LEFT;
        rowc[0].i = exc; rowc[0].j = eyc; rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
        if (exf_local % 2 == 0) {
          nweight = 1;
          if (exf == Nexf || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            weight[0] = 0.375;
          } else {
            weight[0] = 0.25;
          }
        } else {
          nweight = 2;
          rowc[1].i = exc; rowc[1].j = eyc; rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
          weight[0] = 0.125; weight[1] = 0.125;
        }
        ierr = DMStagStencilToIndexLocal(dmc,nweight,rowc,ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Down edges (excluding right "extra" dummy col) */
      if (exf < startexf+nexf) {
        DMStagStencil rowc[2],colf;
        PetscInt      ir[2],ic,nweight;
        PetscScalar   weight[2];
        colf.i    = exf; colf.j    = eyf; colf.c = 0; colf.loc       = DMSTAG_DOWN;
        rowc[0].i = exc; rowc[0].j = eyc; rowc[0].c = 0; rowc[0].loc = DMSTAG_DOWN;
        if (eyf_local % 2 == 0) {
          nweight = 1;
          if (eyf == Neyf || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            weight[0] = 0.375;
          } else {
            weight[0] = 0.25;
          }
        } else {
          nweight = 2;
          rowc[1].i = exc; rowc[1].j = eyc; rowc[1].c = 0; rowc[1].loc = DMSTAG_UP;
          weight[0] = 0.125; weight[1] = 0.125;
        }
        ierr = DMStagStencilToIndexLocal(dmc,nweight,rowc,ir);CHKERRQ(ierr);
        ierr = DMStagStencilToIndexLocal(dmf,1,&colf,&ic);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}
