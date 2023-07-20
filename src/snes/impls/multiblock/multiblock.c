#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscdmplex.h>

#include <petsc/private/dmimpl.h> // For adding dsIn
#include <petsc/private/tsimpl.h> // For DMTS and ARKIMEX copying

PETSC_EXTERN PetscErrorCode DMCopyDMTS(DM, DM);

typedef struct _BlockDesc *BlockDesc;
struct _BlockDesc {
  char     *name;      // Block name
  PetscInt  Nf;        // Number of DM fields
  PetscInt *fields;    // DM fields numbers, or NULL
  char     *labelname; // The label name, or NULL
  PetscInt  labelval;  // The label value
  IS        is;        // Index set defining the block
  SNES      snes;      // subSNES for this block
  TS        ts;        // subTS in case the SNES in embedded in a TS loop
  BlockDesc next, previous;
};

typedef struct {
  PetscBool       defined;        // Flag is true after the blocks have been defined, no more can be added
  PetscBool       setfromoptions; // Flag is true if options were set on this SNES
  PCCompositeType type;           // Solver combination method (additive, multiplicative, etc.)
  PetscInt        Nb;             // Number of blocks
  BlockDesc       blocks;         // Linked list of block descriptors
  PetscBool       useLabel;       // Divide by label instead of fields
} SNES_Multiblock;

static PetscErrorCode SNESReset_Multiblock(SNES snes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *)snes->data;
  BlockDesc        blocks = mb->blocks, next;

  PetscFunctionBegin;
  while (blocks) {
    next = blocks->next;
    PetscCall(ISDestroy(&blocks->is));
    PetscCall(SNESReset(blocks->snes));
    blocks = next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SNESDestroy_Multiblock - Destroys the private SNES_Multiblock context that was created with SNESCreate_Multiblock().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
static PetscErrorCode SNESDestroy_Multiblock(SNES snes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *)snes->data;
  BlockDesc        blocks = mb->blocks, next;

  PetscFunctionBegin;
  PetscCall(SNESReset_Multiblock(snes));
  while (blocks) {
    next = blocks->next;
    PetscCall(PetscFree(blocks->name));
    PetscCall(PetscFree(blocks->fields));
    PetscCall(PetscFree(blocks->labelname));
    PetscCall(SNESDestroy(&blocks->snes));
    PetscCall(TSDestroy(&blocks->ts));
    PetscCall(PetscFree(blocks));
    blocks = next;
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockAddBlock_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockAddDomainBlock_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockGetSubSNES_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockSchurPrecondition_C", NULL));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockSetBlocksFromOptions_Private(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;
  DM               dm;
  PetscInt        *ifields;
  PetscInt         Nf, i;
  PetscBool        flg;
  char             optionname[PETSC_MAX_PATH_LEN], name[8];

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscMalloc1(Nf, &ifields));
  for (i = 0;; ++i) {
    PetscInt nfields = Nf;

    PetscCall(PetscSNPrintf(name, sizeof(name), "%" PetscInt_FMT, i));
    PetscCall(PetscSNPrintf(optionname, sizeof(optionname), "-snes_multiblock_%" PetscInt_FMT "_fields", i));
    PetscCall(PetscOptionsGetIntArray(NULL, ((PetscObject)snes)->prefix, optionname, ifields, &nfields, &flg));
    if (!flg) {
      char    *domain[2];
      PetscInt nmax = 2, val;

      PetscCall(PetscSNPrintf(optionname, sizeof(optionname), "-snes_multiblock_%" PetscInt_FMT "_domain", i));
      PetscCall(PetscOptionsGetStringArray(NULL, ((PetscObject)snes)->prefix, optionname, domain, &nmax, &flg));
      if (!flg) break;
      PetscCheck(nmax == 2, PETSC_COMM_SELF, PETSC_ERR_USER, "Must give labelname,value not %s", domain[0]);
      PetscCall(PetscOptionsStringToInt(domain[1], &val));
      PetscCall(SNESMultiblockAddDomainBlock(snes, name, domain[0], val));
      mb->useLabel = PETSC_TRUE;
    } else {
      PetscCheck(nfields, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot give zero fields for option %s", optionname);
      PetscCall(SNESMultiblockAddBlock(snes, name, nfields, ifields));
    }
  }
  PetscCall(PetscFree(ifields));
  if (i > 0) {
    /* Makes command-line setting of blocks take precedence over setting them in code.
       Otherwise subsequent calls to SNESMultiblockAddBlock() would create new blocks,
       which would probably not be what the user wanted. */
    mb->defined = PETSC_TRUE;
    PetscCall(PetscInfo(snes, "SNESMultiblock blocks defined using the options database\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockSetDefaults(SNES snes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *)snes->data;
  BlockDesc        blocks = mb->blocks;
  DM               dm;
  PetscInt         Nf;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetNumFields(dm, &Nf));
  if (!mb->defined && mb->setfromoptions) PetscCall(SNESMultiblockSetBlocksFromOptions_Private(snes));
  if (mb->Nb == 0)
    for (PetscInt f = 0; f < Nf; ++f) PetscCall(SNESMultiblockAddBlock(snes, NULL, 1, &f));
  else if (!mb->useLabel && mb->Nb == 1) {
    PetscInt n = 0, *fields;

    for (PetscInt f = 0; f < Nf; ++f) {
      PetscBool found = PETSC_FALSE;
      for (PetscInt i = 0; i < blocks->Nf; ++i)
        if (blocks->fields[i] == f) {
          found = PETSC_TRUE;
          break;
        }
      if (!found) ++n;
    }
    PetscCall(PetscMalloc1(n, &fields));
    n = 0;
    for (PetscInt f = 0; f < Nf; ++f) {
      PetscBool found = PETSC_FALSE;
      for (PetscInt i = 0; i < blocks->Nf; ++i)
        if (blocks->fields[i] == f) {
          found = PETSC_TRUE;
          break;
        }
      if (!found) fields[n++] = f;
    }
    PetscCall(SNESMultiblockAddBlock(snes, NULL, n, fields));
    PetscCall(PetscFree(fields));
  }
  PetscCheck(mb->Nb >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNESMultiblock: Must have at least one block, not %" PetscInt_FMT, mb->Nb);
  mb->defined = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCopyToSubTS_Private(TS ts, IS is, TS subts)
{
  DM        subdm;
  Vec       X, Xdot;
  PetscBool isbeuler, istheta, isimex;

  PetscFunctionBegin;
  PetscCall(TSCopy(ts, subts));
  PetscCall(TSSetUp(subts));
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSBEULER, &isbeuler));
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSTHETA, &istheta));
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSARKIMEX, &isimex));
  PetscCall(TSGetDM(subts, &subdm));
  PetscCall(TSGetSolution(ts, &X));
  PetscCall(VecDuplicate(X, &Xdot));
  PetscCall(DMSetAuxiliaryVec(subdm, NULL, 0, 1025, Xdot));
  PetscCall(VecDestroy(&Xdot));
  // Eventually do this with the restrict hook
  if (isbeuler || istheta) {
    DM  dm, subdm;
    Vec X0, subX0;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(TSGetDM(subts, &subdm));
    PetscCall(TSThetaGetX0AndXdot(ts, dm, &X0, NULL));
    PetscCall(TSThetaGetX0AndXdot(subts, subdm, &subX0, NULL));
    PetscCall(VecISCopy(X0, is, SCATTER_REVERSE, subX0));
    PetscCall(TSThetaRestoreX0AndXdot(ts, dm, &X0, NULL));
    PetscCall(TSThetaRestoreX0AndXdot(subts, subdm, &subX0, NULL));
  } else if (isimex) {
    DM  dm, subdm;
    Vec Z, subZ;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(TSGetDM(subts, &subdm));
    PetscCall(TSARKIMEXGetVecs(ts, dm, &Z, NULL));
    PetscCall(TSARKIMEXGetVecs(subts, subdm, &subZ, NULL));
    PetscCall(VecISCopy(Z, is, SCATTER_REVERSE, subZ));
    PetscCall(TSARKIMEXRestoreVecs(ts, dm, &Z, NULL));
    PetscCall(TSARKIMEXRestoreVecs(subts, subdm, &subZ, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockResetDM_Private(SNES snes, DM dm)
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;
  BlockDesc        blocks;
  DMTS             tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTS(dm, &tdm));
  blocks = mb->blocks;
  while (blocks) {
    DM          subdm;
    const char *prefix;

    if (blocks->fields) {
      PetscCall(DMCreateSubDM(dm, blocks->Nf, blocks->fields, &blocks->is, &subdm));
    } else {
      DMLabel label;

      PetscCall(DMGetLabel(dm, blocks->labelname, &label));
      PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Could not find label %s in DM", blocks->labelname);
      PetscCall(DMPlexFilter(dm, label, blocks->labelval, &subdm));
      PetscCall(DMCopyDisc(dm, subdm));
      PetscCall(DMCreateSubDomainIS(dm, subdm, &blocks->is));
      if (PetscDefined(USE_DEBUG)) {
        PetscSection gs;
        PetscInt     n, n2;

        PetscCall(DMGetGlobalSection(subdm, &gs));
        PetscCall(ISGetLocalSize(blocks->is, &n));
        PetscCall(PetscSectionGetConstrainedStorageSize(gs, &n2));
        PetscCheck(n == n2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%" PetscInt_FMT " != %" PetscInt_FMT, n, n2);
      }
    }
    // Turn off preallocation in Plex
    PetscCall(DMSetMatrixPreallocateSkip(subdm, PETSC_TRUE));
    PetscCall(SNESSetDM(blocks->snes, subdm));
    {
      // Set dsIn with superDS in prob
      subdm->probs[0].dsIn = dm->probs[0].ds;
      PetscCall(PetscObjectReference((PetscObject)subdm->probs[0].dsIn));
    }
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)blocks->snes, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)subdm, prefix));
    PetscCall(DMViewDSFromOptions(subdm, NULL, "-dm_petscds_view"));

    PetscCall(DMCopyDMSNES(dm, subdm));
    // This SNES is embedded in a TS
    if (tdm) {
      PetscErrorCode (*func)(SNES, Vec, Vec, void *);
      PetscErrorCode (*jac)(SNES, Vec, Mat, Mat, void *);
      void *ctx;

      PetscCall(DMTSCreateSubDMTS(dm, subdm));
      // If we are in a TS, that TS is the context for DMSNES computefunction and computejacobian
      PetscCall(DMSNESGetFunction(subdm, &func, &ctx));
      PetscCall(DMSNESGetJacobian(subdm, &jac, &ctx));
      PetscCall(TSCreate(PetscObjectComm((PetscObject)ctx), &blocks->ts));
      PetscCall(TSSetDM(blocks->ts, subdm));
      PetscCall(TSSetSNES(blocks->ts, blocks->snes));
      //PetscCall(TSCopyToSubTS_Private((TS)ctx, blocks->is, blocks->ts));
      PetscCall(DMSNESSetFunction(subdm, func, blocks->ts));
      PetscCall(DMSNESSetJacobian(subdm, jac, blocks->ts));
    }
    if (mb->setfromoptions) {
      PetscCall(DMSetFromOptions(subdm));
      PetscCall(SNESSetFromOptions(blocks->snes));
    }
    PetscCall(DMDestroy(&subdm));
    PetscCall(SNESSetUp(blocks->snes));
    blocks = blocks->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetUp_Multiblock(SNES snes)
{
  DM dm;

  PetscFunctionBegin;
  PetscCall(SNESMultiblockSetDefaults(snes));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(SNESMultiblockResetDM_Private(snes, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SNESSetFromOptions_Multiblock - Sets various parameters for the SNESMULTIBLOCK method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_Multiblock(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;
  PCCompositeType  ctype;
  PetscBool        flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNES Multiblock options");
  PetscCall(PetscOptionsEnum("-snes_multiblock_type", "Type of composition", "PCFieldSplitSetType", PCCompositeTypes, (PetscEnum)mb->type, (PetscEnum *)&ctype, &flg));
  if (flg) PetscCall(SNESMultiblockSetType(snes, ctype));
  PetscOptionsHeadEnd();
  mb->setfromoptions = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESView_Multiblock(SNES snes, PetscViewer viewer)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *)snes->data;
  BlockDesc        blocks = mb->blocks;
  PetscBool        iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "SNES Multiblock with %s composition: total blocks = %" PetscInt_FMT "\n", PCCompositeTypes[mb->type], mb->Nb));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Solver info for each split is in the following SNES objects:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    while (blocks) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Block %s", blocks->name));
      if (blocks->fields) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " Fields "));
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
        for (PetscInt j = 0; j < blocks->Nf; ++j) {
          if (j > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ","));
          PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, blocks->fields[j]));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, " Domain %s (%" PetscInt_FMT ")\n", blocks->labelname, blocks->labelval));
      }
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      PetscCall(SNESView(blocks->snes, viewer));
      blocks = blocks->next;
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSolve_Multiblock(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;
  Vec              X, F;
  PetscReal        fnorm;
  PetscInt         maxit;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  snes->reason = SNES_CONVERGED_ITERATING;

  X = snes->vec_sol;  // X^n
  F = snes->vec_func; // residual
  {
    BlockDesc blocks = mb->blocks;
    DM        dm;
    TS        ts;

    PetscCall(SNESGetDM(snes, &dm));
    PetscCall(DMSNESGetFunction(dm, NULL, (void **)&ts));
    while (blocks) {
      DM  dm;
      Vec tmpX;

      PetscCall(TSCopyToSubTS_Private(ts, blocks->is, blocks->ts));
      // Set the solution as an auxiliary vec on the subdm with a special key
      PetscCall(SNESGetDM(blocks->snes, &dm));
      PetscCall(VecDuplicate(X, &tmpX));
      PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 1024, tmpX));
      PetscCall(DMSetSubdofIS(dm, blocks->is));
      blocks = blocks->next;
    }
  }

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  if (!snes->vec_func_init_set) {
    PetscCall(SNESComputeFunction(snes, X, F));
  } else snes->vec_func_init_set = PETSC_FALSE;

  PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
  SNESCheckFunctionNorm(snes, fnorm);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));

  /* test convergence */
  PetscCall(SNESConverged(snes, 0, 0.0, 0.0, fnorm));
  PetscCall(SNESMonitor(snes, 0, fnorm));
  if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(SNESGetTolerances(snes, NULL, NULL, NULL, &maxit, NULL));
  for (PetscInt i = 0; i < maxit; ++i) {
    /* Call general purpose update function */
    PetscTryTypeMethod(snes, update, snes->iter);
    // Update our copy of X^n
    {
      DM  dm;
      Vec tmpX;

      PetscCall(SNESGetDM(mb->blocks->snes, &dm));
      PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 1024, &tmpX));
      PetscCall(VecCopy(X, tmpX));
    }
    /* Compute X^{new} from subsolves */
    if (mb->type == PC_COMPOSITE_MULTIPLICATIVE) {
      BlockDesc   blocks = mb->blocks;
      PetscInt    b      = 0;
      const char *name;

      PetscCall(PetscObjectGetName((PetscObject)X, &name));
      while (blocks) {
        DM  dm;
        Vec u;

        if (b >= 0) {
          PetscCall(SNESGetDM(blocks->snes, &dm));
          PetscCall(DMGetGlobalVector(dm, &u));
          PetscCall(PetscObjectSetName((PetscObject)u, "sol"));
          PetscCall(VecISCopy(X, blocks->is, SCATTER_REVERSE, u));
          PetscCall(PetscObjectSetName((PetscObject)X, "BEFORE Solution"));
          PetscCall(VecViewFromOptions(X, NULL, "-multiblock_sol_view"));
          PetscCall(VecLockReadPush(X));
          // SOMETHING is changing u here
          PetscCall(SNESSolve(blocks->snes, NULL, u));
          PetscCall(VecLockReadPop(X));
          PetscCall(VecISCopy(X, blocks->is, SCATTER_FORWARD, u));
          PetscCall(PetscObjectSetName((PetscObject)X, "AFTER Solution"));
          PetscCall(VecViewFromOptions(X, NULL, "-multiblock_sol_view"));
          PetscCall(DMRestoreGlobalVector(dm, &u));
        }
        ++b;
        blocks = blocks->next;
      }
      PetscCall(PetscObjectSetName((PetscObject)X, name));
    } else SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Unsupported or unknown composition %d", (int)mb->type);

    {
      DM dm, tdm;
      TS ts;

      PetscCall(SNESGetDM(snes, &dm));
      PetscCall(DMSNESGetFunction(dm, NULL, (void **)&ts));
      PetscCall(TSGetDM(ts, &tdm));
      PetscCheck(dm == tdm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "How the hell are these different?");
    }
    // Compute F(X^{new})
    PetscCall(SNESComputeFunction(snes, X, F));
    PetscCall(VecNorm(F, NORM_2, &fnorm));
    SNESCheckFunctionNorm(snes, fnorm);

    /* Monitor convergence */
    if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i + 1;
    snes->norm = fnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes, snes->norm, 0));
    /* Test for convergence */
    PetscCall(SNESConverged(snes, snes->iter, 0.0, 0.0, fnorm));
    PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
    if (snes->reason) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockAddBlock_Multiblock(SNES snes, const char name[], PetscInt n, const PetscInt fields[])
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;
  BlockDesc        newblock, next = mb->blocks;
  DM               dm;
  PetscInt         Nf;
  char             prefix[128];

  PetscFunctionBegin;
  if (mb->defined) {
    PetscCall(PetscInfo(snes, "Ignoring new block \"%s\" because the blocks have already been defined\n", name));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetNumFields(dm, &Nf));
  for (PetscInt i = 0; i < n; ++i) PetscCheck(fields[i] >= 0 && fields[i] < Nf, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid field %" PetscInt_FMT " requested, not in [0, %" PetscInt_FMT ")", fields[i], Nf);
  PetscCall(PetscNew(&newblock));
  if (name) {
    PetscCall(PetscStrallocpy(name, &newblock->name));
  } else {
    PetscInt len = floor(log10(PetscMax(1, mb->Nb))) + 1;

    PetscCall(PetscMalloc1(len + 1, &newblock->name));
    PetscCall(PetscSNPrintf(newblock->name, len + 1, "%" PetscInt_FMT, mb->Nb));
  }
  newblock->Nf = n;

  PetscCall(PetscMalloc1(n, &newblock->fields));
  PetscCall(PetscArraycpy(newblock->fields, fields, n));

  newblock->labelname = NULL;
  newblock->labelval  = -1;
  newblock->next      = NULL;

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)snes), &newblock->snes));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)newblock->snes, (PetscObject)snes, 1));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%smultiblock_%s_", ((PetscObject)snes)->prefix ? ((PetscObject)snes)->prefix : "", newblock->name));
  PetscCall(SNESSetOptionsPrefix(newblock->snes, prefix));

  if (!next) {
    mb->blocks         = newblock;
    newblock->previous = NULL;
  } else {
    while (next->next) next = next->next;
    next->next         = newblock;
    newblock->previous = next;
  }
  mb->Nb++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockAddDomainBlock_Multiblock(SNES snes, const char name[], const char labelname[], PetscInt val)
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;
  BlockDesc        newblock, next = mb->blocks;
  DM               dm;
  char             prefix[128];

  PetscFunctionBegin;
  if (mb->defined) {
    PetscCall(PetscInfo(snes, "Ignoring new block \"%s\" because the blocks have already been defined\n", name));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(PetscNew(&newblock));
  if (name) {
    PetscCall(PetscStrallocpy(name, &newblock->name));
  } else {
    PetscInt len = floor(log10(PetscMax(1, mb->Nb))) + 1;

    PetscCall(PetscMalloc1(len + 1, &newblock->name));
    PetscCall(PetscSNPrintf(newblock->name, len + 1, "%" PetscInt_FMT, mb->Nb));
  }
  newblock->Nf     = 0;
  newblock->fields = NULL;
  newblock->next   = NULL;

  PetscCall(PetscStrallocpy(labelname, &newblock->labelname));
  newblock->labelval = val;

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)snes), &newblock->snes));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)newblock->snes, (PetscObject)snes, 1));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%smultiblock_%s_", ((PetscObject)snes)->prefix ? ((PetscObject)snes)->prefix : "", newblock->name));
  PetscCall(SNESSetOptionsPrefix(newblock->snes, prefix));

  if (!next) {
    mb->blocks         = newblock;
    newblock->previous = NULL;
  } else {
    while (next->next) next = next->next;
    next->next         = newblock;
    newblock->previous = next;
  }
  mb->Nb++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockGetSubSNES_Multiblock(SNES snes, PetscInt *n, SNES **subsnes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *)snes->data;
  BlockDesc        blocks = mb->blocks;
  PetscInt         cnt    = 0;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(mb->Nb, subsnes));
  while (blocks) {
    (*subsnes)[cnt++] = blocks->snes;
    blocks            = blocks->next;
  }
  PetscCheck(cnt == mb->Nb, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Corrupt SNESMULTIBLOCK object: number of blocks in linked list %" PetscInt_FMT " does not match number in object %" PetscInt_FMT, cnt, mb->Nb);

  if (n) *n = mb->Nb;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMultiblockSetType_Multiblock(SNES snes, PCCompositeType type)
{
  SNES_Multiblock *mb = (SNES_Multiblock *)snes->data;

  PetscFunctionBegin;
  mb->type = type;
  if (type == PC_COMPOSITE_SCHUR) {
#if 1
    SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "The Schur composite type is not yet supported");
#else
    snes->ops->solve = SNESSolve_Multiblock_Schur;
    snes->ops->view  = SNESView_Multiblock_Schur;

    PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockGetSubSNES_C", SNESMultiblockGetSubSNES_Schur));
    PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockSchurPrecondition_C", SNESMultiblockSchurPrecondition_Default));
#endif
  } else {
    snes->ops->solve = SNESSolve_Multiblock;
    snes->ops->view  = SNESView_Multiblock;

    PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockGetSubSNES_C", SNESMultiblockGetSubSNES_Multiblock));
    PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockSchurPrecondition_C", NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESMultiblockAddBlock - Sets the fields for one particular block in a `SNESMULTBLOCK` solver

  Logically Collective

  Input Parameters:
+ snes   - the solver
. name   - name of this block, if NULL the number of the block is used
. n      - the number of fields in this block
- fields - the fields in this block

  Level: intermediate

  Note:
  The `SNESMultiblockAddBlock()` is for defining blocks as a group of fields in a DM.
  This function is called once per block (it creates a new block each time). Solve options
  for this block will be available under the prefix -multiblock_BLOCKNAME_.

.seealso: `SNESMULTBLOCK`, `SNESMultiblockGetSubSNES()`, `SNESMULTIBLOCK`
@*/
PetscErrorCode SNESMultiblockAddBlock(SNES snes, const char name[], PetscInt n, const PetscInt *fields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (name) PetscAssertPointer(name, 2);
  PetscCheck(n >= 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "Provided number of fields %" PetscInt_FMT " in split \"%s\" not positive", n, name);
  PetscAssertPointer(fields, 4);
  PetscTryMethod(snes, "SNESMultiblockAddBlock_C", (SNES, const char[], PetscInt, const PetscInt *), (snes, name, n, fields));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESMultiblockAddDomainBlock - Sets the domain for one particular block in a `SNESMULTBLOCK` solver

  Logically Collective

  Input Parameters:
+ snes      - the solver
. name      - name of this block, if NULL the number of the block is used
. labelname - the label name defining the domain
- val       - the label value defining the domain

  Level: intermediate

  Note:
  The `SNESMultiblockAddBlock()` is for defining blocks as a group of fields in a DM.
  This function is called once per block (it creates a new block each time). Solve options
  for this block will be available under the prefix -multiblock_BLOCKNAME_.

.seealso: `SNESMULTBLOCK`, `SNESMultiblockGetSubSNES()`, `SNESMULTIBLOCK`
@*/
PetscErrorCode SNESMultiblockAddDomainBlock(SNES snes, const char name[], const char labelname[], PetscInt val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (name) PetscAssertPointer(name, 2);
  PetscAssertPointer(labelname, 3);
  PetscTryMethod(snes, "SNESMultiblockAddDomainBlock_C", (SNES, const char[], const char[], PetscInt), (snes, name, labelname, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESMultiblockSetType - Sets the type of block combination used for a `SNESMULTBLOCK` solver

  Logically Collective

  Input Parameters:
+ snes - the solver context
- type - `PC_COMPOSITE_ADDITIVE`, `PC_COMPOSITE_MULTIPLICATIVE` (default), `PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE`

  Options Database Key:
. -snes_multiblock_type <type: one of multiplicative, additive, symmetric_multiplicative> - Sets block combination type

  Level: advanced

.seealso: `SNESMULTBLOCK`, `PCCompositeSetType()`, `PC_COMPOSITE_ADDITIVE`, `PC_COMPOSITE_MULTIPLICATIVE`, `PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE`
@*/
PetscErrorCode SNESMultiblockSetType(SNES snes, PCCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscTryMethod(snes, "SNESMultiblockSetType_C", (SNES, PCCompositeType), (snes, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESMultiblockGetSubSNES - Gets the `SNES` contexts for all blocks in a `SNESMULTBLOCK` solver.

  Not Collective but each `SNES` obtained is parallel

  Input Parameter:
. snes - the solver context

  Output Parameters:
+ n       - the number of blocks
- subsnes - the array of `SNES` contexts

  Level: advanced

  Note:
  After `SNESMultiblockGetSubSNES()` the array of `SNES`s MUST be freed by the user
  (not each `SNES`, just the array that contains them).

  You must call `SNESSetUp()` before calling `SNESMultiblockGetSubSNES()`.

.seealso: `SNESMULTBLOCK`, `SNESMultiblockAddBlock()`
@*/
PetscErrorCode SNESMultiblockGetSubSNES(SNES snes, PetscInt *n, SNES *subsnes[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (n) PetscAssertPointer(n, 2);
  PetscUseMethod(snes, "SNESMultiblockGetSubSNES_C", (SNES, PetscInt *, SNES **), (snes, n, subsnes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetDM_Multiblock(SNES snes, DM dm)
{
  PetscFunctionBegin;
  PetscCall(SNESMultiblockResetDM_Private(snes, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  SNESMULTIBLOCK - Multiblock nonlinear solver that can use overlapping or nonoverlapping blocks, organized
  additively (Jacobi) or multiplicatively (Gauss-Seidel).

  Level: beginner

.seealso: `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONLS`, `SNESNEWTONTR`, `SNESNRICHARDSON`, `SNESMultiblockSetType()`,
          `PC_COMPOSITE_ADDITIVE`, `PC_COMPOSITE_MULTIPLICATIVE`, `PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Multiblock(SNES snes)
{
  SNES_Multiblock *mb;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_Multiblock;
  snes->ops->setup          = SNESSetUp_Multiblock;
  snes->ops->setfromoptions = SNESSetFromOptions_Multiblock;
  snes->ops->view           = SNESView_Multiblock;
  snes->ops->solve          = SNESSolve_Multiblock;
  snes->ops->reset          = SNESReset_Multiblock;
  snes->ops->setdm          = SNESSetDM_Multiblock;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&mb));
  snes->data         = (void *)mb;
  mb->defined        = PETSC_FALSE;
  mb->setfromoptions = PETSC_FALSE;
  mb->type           = PC_COMPOSITE_MULTIPLICATIVE;
  mb->Nb             = 0;

  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockAddBlock_C", SNESMultiblockAddBlock_Multiblock));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockAddDomainBlock_C", SNESMultiblockAddDomainBlock_Multiblock));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockSetType_C", SNESMultiblockSetType_Multiblock));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESMultiblockGetSubSNES_C", SNESMultiblockGetSubSNES_Multiblock));
  PetscFunctionReturn(PETSC_SUCCESS);
}
