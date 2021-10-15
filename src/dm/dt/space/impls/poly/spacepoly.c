#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Polynomial(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace polynomial options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_tensor", "Use the tensor product polynomials", "PetscSpacePolynomialSetTensor", poly->tensor, &poly->tensor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(v, "%s space of degree %D\n", poly->tensor ? "Tensor polynomial" : "Polynomial", sp->degree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Polynomial(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePolynomialView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(poly->degrees);CHKERRQ(ierr);
  if (poly->subspaces) {
    PetscInt d;

    for (d = 0; d < sp->Nv; ++d) {
      ierr = PetscSpaceDestroy(&poly->subspaces[d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(poly->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(poly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  PetscInt         ndegree = sp->degree+1;
  PetscInt         deg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (poly->setupCalled) PetscFunctionReturn(0);
  if (sp->Nv <= 1) {
    poly->tensor = PETSC_FALSE;
  }
  if (poly->tensor) {
    sp->maxDegree = PETSC_DETERMINE;
    ierr = PetscSpaceSetType(sp, PETSCSPACETENSOR);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(ndegree, &poly->degrees);CHKERRQ(ierr);
  for (deg = 0; deg < ndegree; ++deg) poly->degrees[deg] = deg;
  if (poly->tensor) {
    sp->maxDegree = sp->degree + PetscMax(sp->Nv - 1,0);
  } else {
    sp->maxDegree = sp->degree;
  }
  poly->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* We treat the space as a tensor product of scalar polynomial spaces, so the dimension is multiplied by Nc */
static PetscErrorCode PetscSpaceGetDimension_Polynomial(PetscSpace sp, PetscInt *dim)
{
  PetscInt         deg  = sp->degree;
  PetscInt         n    = sp->Nv;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDTBinomialInt(n + deg, n, dim);CHKERRQ(ierr);
  *dim *= sp->Nc;
  PetscFunctionReturn(0);
}

/*
  p in [0, npoints), i in [0, pdim), c in [0, Nc)

  B[p][i][c] = B[p][i_scalar][c][c]
*/
static PetscErrorCode PetscSpaceEvaluate_Polynomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         Nc      = sp->Nc;
  PetscInt         dim     = sp->Nv;
  PetscInt         Nb, jet, Njet;
  PetscReal       *pScalar;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!poly->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(sp, npoints, points, B, D, H);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscDTBinomialInt(dim + sp->degree, dim, &Nb);CHKERRQ(ierr);
  if (H) {
    jet = 2;
  } else if (D) {
    jet = 1;
  } else {
    jet = 0;
  }
  ierr = PetscDTBinomialInt(dim + jet, dim, &Njet);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Nb * Njet * npoints, MPIU_REAL, &pScalar);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, npoints, points, sp->degree, jet, pScalar);CHKERRQ(ierr);
  if (B) {
    PetscInt p_strl = Nb*Nc*Nc;
    PetscInt b_strl = Nc*Nc;
    PetscInt c_strl = Nc+1;

    PetscInt b_strr = Njet*npoints;
    PetscInt p_strr = 1;

    ierr = PetscArrayzero(B, npoints * Nb * Nc * Nc);CHKERRQ(ierr);
    for (PetscInt c = 0; c < Nc; c++) {
      for (PetscInt b = 0; b < Nb; b++) {
        for (PetscInt p = 0; p < npoints; p++) {
          B[p*p_strl + b*b_strl + c*c_strl] = pScalar[b*b_strr + p*p_strr];
        }
      }
    }
  }
  if (D) {
    PetscInt p_strl = dim*Nb*Nc*Nc;
    PetscInt b_strl = dim*Nc*Nc;
    PetscInt c_strl = dim*(Nc+1);
    PetscInt d_strl = 1;

    PetscInt b_strr = Njet*npoints;
    PetscInt d_strr = npoints;
    PetscInt p_strr = 1;

    ierr = PetscArrayzero(D, npoints * Nb * Nc * Nc * dim);CHKERRQ(ierr);
    for (PetscInt c = 0; c < Nc; c++) {
      for (PetscInt d = 0; d < dim; d++) {
        for (PetscInt b = 0; b < Nb; b++) {
          for (PetscInt p = 0; p < npoints; p++) {
            D[p*p_strl + b*b_strl + c*c_strl + d*d_strl] = pScalar[b*b_strr + (1+d)*d_strr + p*p_strr];
          }
        }
      }
    }
  }
  if (H) {
    PetscInt p_strl  = dim*dim*Nb*Nc*Nc;
    PetscInt b_strl  = dim*dim*Nc*Nc;
    PetscInt c_strl  = dim*dim*(Nc+1);
    PetscInt d1_strl = dim;
    PetscInt d2_strl = 1;

    PetscInt b_strr = Njet*npoints;
    PetscInt j_strr = npoints;
    PetscInt p_strr = 1;

    PetscInt *derivs;
    ierr = PetscCalloc1(dim, &derivs);CHKERRQ(ierr);
    ierr = PetscArrayzero(H, npoints * Nb * Nc * Nc * dim * dim);CHKERRQ(ierr);
    for (PetscInt d1 = 0; d1 < dim; d1++) {
      for (PetscInt d2 = 0; d2 < dim; d2++) {
        PetscInt j;
        derivs[d1]++;
        derivs[d2]++;
        ierr = PetscDTGradedOrderToIndex(dim, derivs, &j);CHKERRQ(ierr);
        derivs[d1]--;
        derivs[d2]--;
        for (PetscInt c = 0; c < Nc; c++) {
          for (PetscInt b = 0; b < Nb; b++) {
            for (PetscInt p = 0; p < npoints; p++) {
              H[p*p_strl + b*b_strl + c*c_strl + d1*d1_strl + d2*d2_strl] = pScalar[b*b_strr + j*j_strr + p*p_strr];
            }
          }
        }
      }
    }
    ierr = PetscFree(derivs);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, Nb * Njet * npoints, MPIU_REAL, &pScalar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialSetTensor - Set whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variable is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
+ sp     - the function space object
- tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Options Database:
. -petscspace_poly_tensor <bool> - Whether to use tensor product polynomials in higher dimension

  Level: intermediate

.seealso: PetscSpacePolynomialGetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace sp, PetscBool tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpacePolynomialSetTensor_C",(PetscSpace,PetscBool),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePolynomialGetTensor - Get whether a function space is a space of tensor polynomials (the space is spanned
  by polynomials whose degree in each variabl is bounded by the given order), as opposed to polynomials (the space is
  spanned by polynomials whose total degree---summing over all variables---is bounded by the given order).

  Input Parameters:
. sp     - the function space object

  Output Parameters:
. tensor - PETSC_TRUE for a tensor polynomial space, PETSC_FALSE for a polynomial space

  Level: intermediate

.seealso: PetscSpacePolynomialSetTensor(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace sp, PetscBool *tensor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  ierr = PetscTryMethod(sp,"PetscSpacePolynomialGetTensor_C",(PetscSpace,PetscBool*),(sp,tensor));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialSetTensor_Polynomial(PetscSpace sp, PetscBool tensor)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  poly->tensor = tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialGetTensor_Polynomial(PetscSpace sp, PetscBool *tensor)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  *tensor = poly->tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetHeightSubspace_Polynomial(PetscSpace sp, PetscInt height, PetscSpace *subsp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         Nc, dim, order;
  PetscBool        tensor;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(sp, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &order, NULL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(sp, &tensor);CHKERRQ(ierr);
  if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for space at height %D for dimension %D space", height, dim);
  if (!poly->subspaces) {ierr = PetscCalloc1(dim, &poly->subspaces);CHKERRQ(ierr);}
  if (height <= dim) {
    if (!poly->subspaces[height-1]) {
      PetscSpace  sub;
      const char *name;

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) sp), &sub);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) sp,  &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) sub,  name);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(sub, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(sub, Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(sub, order, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(sub, dim-height);CHKERRQ(ierr);
      ierr = PetscSpacePolynomialSetTensor(sub, tensor);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(sub);CHKERRQ(ierr);
      poly->subspaces[height-1] = sub;
    }
    *subsp = poly->subspaces[height-1];
  } else {
    *subsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Polynomial(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscSpaceSetFromOptions_Polynomial;
  sp->ops->setup             = PetscSpaceSetUp_Polynomial;
  sp->ops->view              = PetscSpaceView_Polynomial;
  sp->ops->destroy           = PetscSpaceDestroy_Polynomial;
  sp->ops->getdimension      = PetscSpaceGetDimension_Polynomial;
  sp->ops->evaluate          = PetscSpaceEvaluate_Polynomial;
  sp->ops->getheightsubspace = PetscSpaceGetHeightSubspace_Polynomial;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", PetscSpacePolynomialGetTensor_Polynomial);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialSetTensor_C", PetscSpacePolynomialSetTensor_Polynomial);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOLYNOMIAL = "poly" - A PetscSpace object that encapsulates a polynomial space, e.g. P1 is the space of
  linear polynomials. The space is replicated for each component.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&poly);CHKERRQ(ierr);
  sp->data = poly;

  poly->tensor       = PETSC_FALSE;
  poly->degrees      = NULL;
  poly->subspaces    = NULL;

  ierr = PetscSpaceInitialize_Polynomial(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

