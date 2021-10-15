#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Ptrimmed(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace polynomial options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_ptrimmed_form_degree", "form degree of trimmed space", "PetscSpacePTrimmedSetFormDegree", pt->formDegree, &(pt->formDegree), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             f;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  f = pt->formDegree;
  ierr = PetscViewerASCIIPrintf(v, "Trimmed polynomials %D%s-forms of degree %D\n", PetscAbsInt(f), f < 0 ? "*" : "", sp->maxDegree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Ptrimmed(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePTrimmedView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedGetFormDegree_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedSetFormDegree_C", NULL);CHKERRQ(ierr);
  if (pt->subspaces) {
    PetscInt d;

    for (d = 0; d < sp->Nv; ++d) {
      ierr = PetscSpaceDestroy(&pt->subspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(pt->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(pt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscSpaceSetUp_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             Nf;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (pt->setupCalled) PetscFunctionReturn(0);
  if (pt->formDegree < -sp->Nv || pt->formDegree > sp->Nv) SETERRQ3(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Form degree %D not in valid range [%D,%D]", pt->formDegree, sp->Nv, sp->Nv);
  ierr = PetscDTBinomialInt(sp->Nv, PetscAbsInt(pt->formDegree), &Nf);CHKERRQ(ierr);
  if (sp->Nc == PETSC_DETERMINE) {
    sp->Nc = Nf;
  }
  if (sp->Nc % Nf) SETERRQ2(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "Number of components %D is not a multiple of form dimension %D\n", sp->Nc, Nf);
  if (sp->degree == PETSC_DEFAULT) {
    sp->degree = 0;
  } else if (sp->degree < 0) {
    SETERRQ1(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Invalid negative degree %D\n", sp->degree);
  }
  if (pt->formDegree == 0 || PetscAbsInt(pt->formDegree) == sp->Nv) {
    // Convert to regular polynomial space
    ierr = PetscSpaceSetType(sp, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  pt->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Ptrimmed(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt             f;
  PetscInt             Nf;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  f = pt->formDegree;
  // For PetscSpace, degree refers to the largest complete polynomial degree contained in the space which
  // is equal to the index of a P trimmed space only for 0-forms: otherwise, the index is degree + 1
  ierr = PetscDTPTrimmedSize(sp->Nv, f == 0 ? sp->degree : sp->degree + 1, pt->formDegree, dim);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(sp->Nv, PetscAbsInt(pt->formDegree), &Nf);CHKERRQ(ierr);
  *dim *= (sp->Nc / Nf);
  PetscFunctionReturn(0);
}

/*
  p in [0, npoints), i in [0, pdim), c in [0, Nc)

  B[p][i][c] = B[p][i_scalar][c][c]
*/
static PetscErrorCode PetscSpaceEvaluate_Ptrimmed(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Ptrimmed *pt    = (PetscSpace_Ptrimmed *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         jet, degree, Nf, Ncopies, Njet;
  PetscInt         Nc      = sp->Nc;
  PetscInt         f;
  PetscInt         dim     = sp->Nv;
  PetscReal       *eval;
  PetscInt         Nb;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (H) {
    jet = 2;
  } else if (D) {
    jet = 1;
  } else {
    jet = 0;
  }
  f = pt->formDegree;
  degree = f == 0 ? sp->degree : sp->degree + 1;
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(f), &Nf);CHKERRQ(ierr);
  Ncopies = Nc / Nf;
  ierr = PetscDTBinomialInt(dim + jet, dim, &Njet);CHKERRQ(ierr);
  ierr = PetscDTPTrimmedSize(dim, degree, f, &Nb);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Nb * Nf * Njet * npoints, MPIU_REAL, &eval);CHKERRQ(ierr);
  ierr = PetscDTPTrimmedEvalJet(dim, npoints, points, degree, f, jet, eval);CHKERRQ(ierr);
  if (B) {
    for (PetscInt c = 0; c < Ncopies; c++) {
      for (PetscInt v = 0; v < Nf; v++) {
        for (PetscInt b = 0; b < Nb; b++) {
          for (PetscInt p = 0; p < npoints; p++) {
            B[p*Nb*Nc + b*Nc + c*Nf + v] = eval[b*Nf*Njet*npoints + v*Njet*npoints + p];
          }
        }
      }
    }
  }
  if (D) {
    for (PetscInt c = 0; c < Ncopies; c++) {
      for (PetscInt v = 0; v < Nf; v++) {
        for (PetscInt d = 0; d < dim; d++) {
          for (PetscInt b = 0; b < Nb; b++) {
            for (PetscInt p = 0; p < npoints; p++) {
              D[p*Nb*Nc*dim + b*Nc*dim + c*Nf*dim + v*dim + d] = eval[b*Nf*Njet*npoints + v*Njet*npoints + (1+d)*npoints + p];
            }
          }
        }
      }
    }
  }
  if (H) {
    PetscInt *derivs;
    ierr = PetscCalloc1(dim, &derivs);CHKERRQ(ierr);
    for (PetscInt d1 = 0; d1 < dim; d1++) {
      for (PetscInt d2 = 0; d2 < dim; d2++) {
        PetscInt j;
        derivs[d1]++;
        derivs[d2]++;
        ierr = PetscDTGradedOrderToIndex(dim, derivs, &j);CHKERRQ(ierr);
        derivs[d1]--;
        derivs[d2]--;
        for (PetscInt c = 0; c < Ncopies; c++) {
          for (PetscInt v = 0; v < Nf; v++) {
            for (PetscInt b = 0; b < Nb; b++) {
              for (PetscInt p = 0; p < npoints; p++) {
                H[p*Nb*Nc*dim*dim + b*Nc*dim*dim + c*Nf*dim*dim + v*dim*dim + d1*dim + d2] = eval[b*Nf*Njet*npoints + v*Njet*npoints + j*npoints + p];
              }
            }
          }
        }
      }
    }
  }
  ierr = DMRestoreWorkArray(dm, Nb * Nf * Njet * npoints, MPIU_REAL, &eval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePTrimmedSetFormDegree - Set the form degree of the trimmed polynomials.

  Input Parameters:
+ sp         - the function space object
- formDegree - the form degree

  Options Database:
. -petscspace_ptrimmed_form_degree <int> - The trimmed polynomial form degree

  Level: intermediate

.seealso: PetscDTAltV, PetscDTPTrimmedEvalJet(), PetscSpacePTrimmedGetFormDegree()
@*/
PetscErrorCode PetscSpacePTrimmedSetFormDegree(PetscSpace sp, PetscInt formDegree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpacePTrimmedSetFormDegree_C",(PetscSpace,PetscInt),(sp,formDegree));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePTrimmedGetFormDegree - Get the form degree of the trimmed polynomials.

  Input Parameters:
. sp     - the function space object

  Output Parameters:
. formDegee - the form degree

  Level: intermediate

.seealso: PetscDTAltV, PetscDTPTrimmedEvalJet(), PetscSpacePTrimmedSetFormDegree()
@*/
PetscErrorCode PetscSpacePTrimmedGetFormDegree(PetscSpace sp, PetscInt *formDegree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(formDegree, 2);
  ierr = PetscTryMethod(sp,"PetscSpacePTrimmedGetFormDegree_C",(PetscSpace,PetscInt*),(sp,formDegree));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedSetFormDegree_Ptrimmed(PetscSpace sp, PetscInt formDegree)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;

  PetscFunctionBegin;
  pt->formDegree = formDegree;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePTrimmedGetFormDegree_Ptrimmed(PetscSpace sp, PetscInt *formDegree)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(formDegree, 2);
  *formDegree = pt->formDegree;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Ptrimmed(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Ptrimmed *pt = (PetscSpace_Ptrimmed *) sp->data;
  PetscInt         dim;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp, &dim);CHKERRQ(ierr);
  if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!pt->subspaces) {ierr = PetscCalloc1(dim, &(pt->subspaces));CHKERRQ(ierr);}
  if ((dim - height) <= PetscAbsInt(pt->formDegree)) {
    if (!pt->subspaces[height-1]) {
      PetscInt Nc, degree, Nf, Ncopies, Nfsub;
      PetscSpace  sub;
      const char *name;

      ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
      ierr = PetscDTBinomialInt(dim, PetscAbsInt(pt->formDegree), &Nf);CHKERRQ(ierr);
      ierr = PetscDTBinomialInt((dim-height), PetscAbsInt(pt->formDegree), &Nfsub);CHKERRQ(ierr);
      Ncopies = Nf / Nc;
      ierr = PetscSpaceGetDegree(sp, &degree, NULL);CHKERRQ(ierr);

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) sp,  &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACEPTRIMMED);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nfsub * Ncopies);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(sub, degree, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      ierr = PetscSpacePTrimmedSetFormDegree(sub, pt->formDegree);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      pt->subspaces[height-1] = sub;
    }
    *subsp = pt->subspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Ptrimmed(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedGetFormDegree_C", PetscSpacePTrimmedGetFormDegree_Ptrimmed);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePTrimmedSetFormDegree_C", PetscSpacePTrimmedSetFormDegree_Ptrimmed);CHKERRQ(ierr);
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Ptrimmed;
  sp->ops->setup             = PetscSpaceSetUp_Ptrimmed;
  sp->ops->view              = PetscSpaceView_Ptrimmed;
  sp->ops->destroy           = PetscSpaceDestroy_Ptrimmed;
  sp->ops->getdimension      = PetscSpaceGetDimension_Ptrimmed;
  sp->ops->evaluate          = PetscSpaceEvaluate_Ptrimmed;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Ptrimmed;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPTRIMMED = "ptrimmed" - A PetscSpace object that encapsulates a trimmed polynomial space.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType(), PetscDTPTrimmedEvalJet()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Ptrimmed(PetscSpace sp)
{
  PetscSpace_Ptrimmed *pt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&pt);CHKERRQ(ierr);
  sp->data = pt;

  pt->subspaces = NULL;
  sp->Nc        = PETSC_DETERMINE;

  ierr = PetscSpaceInitialize_Ptrimmed(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

