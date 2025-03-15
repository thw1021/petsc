#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

static PetscBool PCCarrierCite;
const char       PCCarrierCitation[] = "@Article{carrierheath1022,\n"
                                       "  author        = {Erin Carrier, Michael T. Heath},\n"
                                       "  title         = {Exploiting compression in solving discretized linear systems},\n"
                                       "  journal       = {ETNA},\n"
                                       "  year          = {2022},\n"
                                       "  pages         = {341--364},\n"
                                       "  volume        = {55}\n"
                                       "}\n";
/*MC
  PCCarrier - a nonlinear preconditioner for `KSPFGMRES` that solves with any provided Carrier-Heath {cite}`carrierheath2022` subspace

  Notes:
  This `PC` is used with `KSPFGMRES` only and is not a traditional preconditioner. Run with
.vb
  -pc_type carrier -ksp_pc_side right -ksp_type fgmres
.ve

  There is currently one pre-wired basis; the standard Krylov base, $ { b, Ab, A^2 b, ...} $ which is
  used by default. Note that the  products by $A$ produce a space that is difficult to orthogonalize so
  `-ksp_gmres_modifiedgramschmidt` is likely needed and even the `KSPFGMRES` my start to have trouble; using
  ``--with-precision=__float128`` resolves these problems but is impractical.

  Developer Note:
  This is incomplete and retires a full implementation of `PCCarrierSetType()` etc

.seealso:  `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `KSPFGMRES`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Carrier(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCCarrierSetType(pc, PC_CARRIER_GMRES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Private context (data structure) for CARRIER_GMRES which is a test basis for the PCCARRIER code
*/
typedef struct {
  Vec      Anv;
  PetscInt it;
} PC_Carrier_GMRES;

static PetscErrorCode PCSetUp_GMRES(PC pc)
{
  PC_Carrier_GMRES *jac;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &jac));
  if (!jac->Anv) PetscCall(MatCreateVecs(pc->pmat, &jac->Anv, NULL));
  jac->it = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_GMRES(PC pc, Vec x, Vec y)
{
  PC_Carrier_GMRES *jac;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &jac));
  if (!jac->it) {
    PetscCall(VecCopy(x, y));
  } else {
    PetscCall(MatMult(pc->mat, jac->Anv, y));
  }
  PetscCall(VecCopy(y, jac->Anv));
  jac->it++;
  PetscCall(PetscCitationsRegister(PCCarrierCitation, &PCCarrierCite));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_GMRES(PC pc)
{
  PC_Carrier_GMRES *jac;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &jac));
  PetscCall(VecDestroy(&jac->Anv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_GMRES(PC pc)
{
  PC_Carrier_GMRES *jac;

  PetscFunctionBegin;
  PetscCall(PCReset_GMRES(pc));
  PetscCall(PCShellGetContext(pc, &jac));
  PetscCall(PetscFree(jac));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PCCarrierSetType(PC pc, PCCarrierType type)
{
  PC_Carrier_GMRES *gmres;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gmres));
  PetscCall(PCShellSetContext(pc, gmres));
  PetscCall(PCShellSetSetUp(pc, PCSetUp_GMRES));
  PetscCall(PCShellSetApply(pc, PCApply_GMRES));
  PetscCall(PCShellSetDestroy(pc, PCDestroy_GMRES));
  PetscFunctionReturn(PETSC_SUCCESS);
}
