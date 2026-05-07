/*
     This file implements an AmgX preconditioner in PETSc as part of PC.
 */

/*
   Include files needed for the AmgX preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
#include <petscdevice_cuda.h>
/* amgx_config.h performs arithmetic between different enum types (AMGX_ASSEMBLE_MODE
   macro and AMGX_modeRange initializer), which GCC >= 13 rejects under
   -Werror=deprecated-enum-enum-conversion.  Suppress the warning around the include. */
#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion"
#endif
#include <amgx_c.h>
#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC diagnostic pop
#endif
#include <algorithm>
#include <limits>
#include <map>

enum class AmgXSmoother {
  PCG,
  PCGF,
  PBiCGStab,
  GMRES,
  FGMRES,
  JacobiL1,
  BlockJacobi,
  GS,
  MulticolorGS,
  MulticolorILU,
  MulticolorDILU,
  ChebyshevPoly,
  NoSolver
};
enum class AmgXAMGMethod {
  Classical,
  Aggregation
};
enum class AmgXSelector {
  Size2,
  Size4,
  Size8,
  MultiPairwise,
  PMIS,
  HMIS
};
enum class AmgXCoarseSolver {
  DenseLU,
  NoSolver
};
enum class AmgXAMGCycle {
  V,
  W,
  F,
  CG,
  CGF
};

struct AmgXControlMap {
  static const std::map<std::string, AmgXAMGMethod>    AMGMethods;
  static const std::map<std::string, AmgXSmoother>     Smoothers;
  static const std::map<std::string, AmgXSelector>     Selectors;
  static const std::map<std::string, AmgXCoarseSolver> CoarseSolvers;
  static const std::map<std::string, AmgXAMGCycle>     AMGCycles;
};

const std::map<std::string, AmgXAMGMethod> AmgXControlMap::AMGMethods = {
  {"CLASSICAL",   AmgXAMGMethod::Classical  },
  {"AGGREGATION", AmgXAMGMethod::Aggregation}
};

const std::map<std::string, AmgXSmoother> AmgXControlMap::Smoothers = {
  {"PCG",             AmgXSmoother::PCG           },
  {"PCGF",            AmgXSmoother::PCGF          },
  {"PBICGSTAB",       AmgXSmoother::PBiCGStab     },
  {"GMRES",           AmgXSmoother::GMRES         },
  {"FGMRES",          AmgXSmoother::FGMRES        },
  {"JACOBI_L1",       AmgXSmoother::JacobiL1      },
  {"BLOCK_JACOBI",    AmgXSmoother::BlockJacobi   },
  {"GS",              AmgXSmoother::GS            },
  {"MULTICOLOR_GS",   AmgXSmoother::MulticolorGS  },
  {"MULTICOLOR_ILU",  AmgXSmoother::MulticolorILU },
  {"MULTICOLOR_DILU", AmgXSmoother::MulticolorDILU},
  {"CHEBYSHEV_POLY",  AmgXSmoother::ChebyshevPoly },
  {"NOSOLVER",        AmgXSmoother::NoSolver      }
};

const std::map<std::string, AmgXSelector> AmgXControlMap::Selectors = {
  {"SIZE_2",         AmgXSelector::Size2        },
  {"SIZE_4",         AmgXSelector::Size4        },
  {"SIZE_8",         AmgXSelector::Size8        },
  {"MULTI_PAIRWISE", AmgXSelector::MultiPairwise},
  {"PMIS",           AmgXSelector::PMIS         },
  {"HMIS",           AmgXSelector::HMIS         }
};

const std::map<std::string, AmgXCoarseSolver> AmgXControlMap::CoarseSolvers = {
  {"DENSE_LU_SOLVER", AmgXCoarseSolver::DenseLU },
  {"NOSOLVER",        AmgXCoarseSolver::NoSolver}
};

const std::map<std::string, AmgXAMGCycle> AmgXControlMap::AMGCycles = {
  {"V",   AmgXAMGCycle::V  },
  {"W",   AmgXAMGCycle::W  },
  {"F",   AmgXAMGCycle::F  },
  {"CG",  AmgXAMGCycle::CG },
  {"CGF", AmgXAMGCycle::CGF}
};

/*
   Private context (data structure) for the AMGX preconditioner.
*/
struct PC_AMGX {
  AMGX_solver_handle    solver;
  AMGX_config_handle    cfg;
  AMGX_resources_handle rsrc;
  PetscBool             solve_state_init;
  PetscBool             rsrc_init;
  PetscBool             verbose;

  AMGX_matrix_handle A;
  AMGX_vector_handle sol;
  AMGX_vector_handle rhs;

  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscMPIInt nranks;
  int         devID; /* cudaGetDevice() returns int */

  void       *lib_handle;
  std::string cfg_contents;

  // Cached state for re-setup
  PetscInt           nnz;
  PetscInt           nLocalRows;
  PetscInt           nGlobalRows;
  PetscInt           bSize;
  Mat                localA;
  const PetscScalar *values;

  // AMG Control parameters
  AmgXSmoother     smoother;
  AmgXAMGMethod    amg_method;
  AmgXSelector     selector;
  AmgXCoarseSolver coarse_solver;
  AmgXAMGCycle     amg_cycle;
  PetscInt         presweeps;
  PetscInt         postsweeps;
  PetscInt         max_levels;
  PetscInt         aggressive_levels;
  PetscInt         dense_lu_num_rows;
  PetscScalar      strength_threshold;
  PetscBool        print_grid_stats;
  PetscBool        exact_coarse_solve;

  // Smoother control parameters
  PetscScalar jacobi_relaxation_factor;
  PetscScalar gs_symmetric;
};

static PetscInt s_count = 0;

/* Buffer of messages from AmgX.
   This global is required because the AMGX_register_print_callback API does not
   support a user-data pointer, so the callback cannot capture per-PC state.
   Not thread-safe; acceptable because PETSc is single-threaded at the PC level. */
static std::string amgx_output{};

// A print callback that allows AmgX to return status messages
static void print_callback(const char *msg, int length)
{
  amgx_output.append(msg);
}

// Outputs messages from the AmgX message buffer and clears it
static PetscErrorCode amgx_output_messages(PC_AMGX *amgx)
{
  PetscFunctionBegin;
  // If AmgX output is enabled and we have a message, output it
  if (amgx->verbose && !amgx_output.empty()) {
    // Only a single rank to output the AmgX messages
    PetscCall(PetscPrintf(amgx->comm, "AMGX: %s", amgx_output.c_str()));

    // Note that all ranks clear their received output
    amgx_output.clear();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// XXX Need to add call in AmgX API that gracefully destroys everything
// without abort etc.
#define PetscCallAmgX(rc) \
  do { \
    AMGX_RC err = (rc); \
    char    msg[4096]; \
    switch (err) { \
    case AMGX_RC_OK: \
      break; \
    default: \
      AMGX_get_error_string(err, msg, 4096); \
      SETERRQ(amgx->comm, PETSC_ERR_LIB, "%s", msg); \
    } \
  } while (0)

/*
   PCSetUp_AMGX - Prepares for the use of the AmgX preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Note:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_AMGX(PC pc)
{
  PC_AMGX                 *amgx = (PC_AMGX *)pc->data;
  Mat                      Pmat = pc->pmat;
  PetscBool                is_dev_ptrs, partial_setup_allowed, done;
  const PetscInt          *colIndices, *rowOffsets;
  PetscInt                *partitionOffsets, nlr = 0, i;
  AMGX_distribution_handle dist;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)Pmat, &is_dev_ptrs, MATAIJCUSPARSE, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));

  /* AmgX requires a sequential local matrix; extract it for MPI cases */
  partial_setup_allowed = (PetscBool)(pc->setupcalled && pc->flag != DIFFERENT_NONZERO_PATTERN);
  if (amgx->nranks > 1) {
    if (partial_setup_allowed) {
      PetscCall(MatMPIAIJGetLocalMat(Pmat, MAT_REUSE_MATRIX, &amgx->localA));
    } else {
      PetscCall(MatMPIAIJGetLocalMat(Pmat, MAT_INITIAL_MATRIX, &amgx->localA));
    }

    if (is_dev_ptrs) PetscCall(MatConvert(amgx->localA, MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &amgx->localA));
  } else {
    amgx->localA = Pmat;
  }

  if (is_dev_ptrs) {
    PetscCall(MatSeqAIJCUSPARSEGetArrayRead(amgx->localA, &amgx->values));
  } else {
    PetscCall(MatSeqAIJGetArrayRead(amgx->localA, &amgx->values));
  }

  if (!partial_setup_allowed) {
    if (!amgx->rsrc_init) {
      PetscCallAmgX(AMGX_config_create(&amgx->cfg, amgx->cfg_contents.c_str()));
      PetscCallAmgX(AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &amgx->devID));
      amgx->rsrc_init = PETSC_TRUE;
    }

    PetscCheck(!amgx->solve_state_init, amgx->comm, PETSC_ERR_PLIB, "AmgX solve state initialisation already called.");
    PetscCallAmgX(AMGX_matrix_create(&amgx->A, amgx->rsrc, AMGX_mode_dDDI));
    PetscCallAmgX(AMGX_vector_create(&amgx->sol, amgx->rsrc, AMGX_mode_dDDI));
    PetscCallAmgX(AMGX_vector_create(&amgx->rhs, amgx->rsrc, AMGX_mode_dDDI));
    PetscCallAmgX(AMGX_solver_create(&amgx->solver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg));
    amgx->solve_state_init = PETSC_TRUE;

    PetscCall(MatGetRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &amgx->nLocalRows, &rowOffsets, &colIndices, &done));
    PetscCheck(done, amgx->comm, PETSC_ERR_PLIB, "MatGetRowIJ was not successful");
    PetscCheck(amgx->nLocalRows < std::numeric_limits<int>::max(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "AmgX restricted to int local rows but nLocalRows = %" PetscInt_FMT " > max<int>", amgx->nLocalRows);

    if (is_dev_ptrs) {
      PetscCallCUDA(cudaMemcpy(&amgx->nnz, &rowOffsets[amgx->nLocalRows], sizeof(int), cudaMemcpyDefault));
    } else {
      amgx->nnz = rowOffsets[amgx->nLocalRows];
    }

    PetscCheck(amgx->nnz < std::numeric_limits<int>::max(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support for 64-bit integer nnz not yet implemented, nnz = %" PetscInt_FMT ".", amgx->nnz);

    /* Build partition offsets: partitionOffsets[i+1] = sum of nLocalRows for ranks 0..i */
    PetscCall(PetscMalloc1(amgx->nranks + 1, &partitionOffsets));
    partitionOffsets[0] = 0; /* could use PetscLayoutGetRanges */
    PetscCallMPI(MPI_Allgather(&amgx->nLocalRows, 1, MPIU_INT, partitionOffsets + 1, 1, MPIU_INT, amgx->comm));
    for (i = 1; i <= amgx->nranks; i++) partitionOffsets[i] += partitionOffsets[i - 1];
    amgx->nGlobalRows = partitionOffsets[amgx->nranks];

    PetscCall(MatGetBlockSize(Pmat, &amgx->bSize));

    /* XXX Currently constrained to 32-bit indices, to be changed in the future */
    PetscCallAmgX(AMGX_distribution_create(&dist, amgx->cfg));
    PetscCallAmgX(AMGX_distribution_set_32bit_colindices(dist, true));
    PetscCallAmgX(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partitionOffsets));
    PetscCallAmgX(AMGX_matrix_upload_distributed(amgx->A, amgx->nGlobalRows, (int)amgx->nLocalRows, (int)amgx->nnz, amgx->bSize, amgx->bSize, rowOffsets, colIndices, amgx->values, NULL, dist));
    PetscCallAmgX(AMGX_solver_setup(amgx->solver, amgx->A));
    PetscCallAmgX(AMGX_vector_bind(amgx->sol, amgx->A));
    PetscCallAmgX(AMGX_vector_bind(amgx->rhs, amgx->A));
    PetscCall(PetscFree(partitionOffsets));

    PetscCall(MatRestoreRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &nlr, &rowOffsets, &colIndices, &done));
  } else {
    /* Fast path: sparsity pattern unchanged, only update coefficients */
    PetscCallAmgX(AMGX_matrix_replace_coefficients(amgx->A, amgx->nLocalRows, amgx->nnz, amgx->values, NULL));
    PetscCallAmgX(AMGX_solver_resetup(amgx->solver, amgx->A));
  }

  if (is_dev_ptrs) {
    PetscCall(MatSeqAIJCUSPARSERestoreArrayRead(amgx->localA, &amgx->values));
  } else {
    PetscCall(MatSeqAIJRestoreArrayRead(amgx->localA, &amgx->values));
  }
  PetscCall(amgx_output_messages(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCApply_AMGX - Applies the AmgX preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  b - rhs vector

   Output Parameter:
.  x - solution vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_AMGX(PC pc, Vec b, Vec x)
{
  PC_AMGX           *amgx = (PC_AMGX *)pc->data;
  PetscScalar       *x_;
  const PetscScalar *b_;
  PetscBool          is_dev_ptrs;
  AMGX_SOLVE_STATUS  status;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x, &is_dev_ptrs, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));

  if (is_dev_ptrs) {
    PetscCall(VecCUDAGetArrayWrite(x, &x_));
    PetscCall(VecCUDAGetArrayRead(b, &b_));
  } else {
    PetscCall(VecGetArrayWrite(x, &x_));
    PetscCall(VecGetArrayRead(b, &b_));
  }

  PetscCallAmgX(AMGX_vector_upload(amgx->sol, amgx->nLocalRows, 1, x_));
  PetscCallAmgX(AMGX_vector_upload(amgx->rhs, amgx->nLocalRows, 1, b_));
  PetscCallAmgX(AMGX_solver_solve_with_0_initial_guess(amgx->solver, amgx->rhs, amgx->sol));

  PetscCallAmgX(AMGX_solver_get_status(amgx->solver, &status));
  PetscCall(PCSetErrorIfFailure(pc, (PetscBool)(status == AMGX_SOLVE_FAILED)));
  PetscCheck(status != AMGX_SOLVE_FAILED, amgx->comm, PETSC_ERR_CONV_FAILED, "AmgX solver failed to solve the system! The error code is %d.", status);
  PetscCallAmgX(AMGX_vector_download(amgx->sol, x_));

  if (is_dev_ptrs) {
    PetscCall(VecCUDARestoreArrayWrite(x, &x_));
    PetscCall(VecCUDARestoreArrayRead(b, &b_));
  } else {
    PetscCall(VecRestoreArrayWrite(x, &x_));
    PetscCall(VecRestoreArrayRead(b, &b_));
  }
  PetscCall(amgx_output_messages(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (amgx->solve_state_init) {
    PetscCallAmgX(AMGX_solver_destroy(amgx->solver));
    PetscCallAmgX(AMGX_matrix_destroy(amgx->A));
    PetscCallAmgX(AMGX_vector_destroy(amgx->sol));
    PetscCallAmgX(AMGX_vector_destroy(amgx->rhs));
    if (amgx->nranks > 1) PetscCall(MatDestroy(&amgx->localA));
    PetscCall(amgx_output_messages(amgx));
    amgx->solve_state_init = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCDestroy_AMGX - Destroys the private context for the AmgX preconditioner
   that was created with PCCreate_AMGX().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_AMGX(pc));
  /* decrease the number of instances; only the last instance destroys resources and finalizes AmgX */
  if (s_count == 1) {
    /* can put this in a PCAMGXInitializePackage method */
    PetscCheck(amgx->rsrc != nullptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "s_rsrc == NULL");
    PetscCallAmgX(AMGX_resources_destroy(amgx->rsrc));
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
    PetscCallAmgX(AMGX_finalize_plugins());
    PetscCallAmgX(AMGX_finalize());
    PetscCallMPI(MPI_Comm_free(&amgx->comm));
  } else {
    PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
  }
  s_count -= 1;
  PetscCall(PetscFree(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Return the key in map whose value equals key, or "" if not found */
template <class T>
std::string map_reverse_lookup(const std::map<std::string, T> &map, const T &key)
{
  for (auto const &m : map) {
    if (m.second == key) return m.first;
  }
  return "";
}

static PetscErrorCode PCSetFromOptions_AMGX(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_AMGX      *amgx          = (PC_AMGX *)pc->data;
  constexpr int MAX_PARAM_LEN = 128;
  char          option[MAX_PARAM_LEN];

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "AmgX options");
  amgx->cfg_contents = "config_version=2,";
  amgx->cfg_contents += "determinism_flag=1,";

  PetscCall(PetscOptionsBool("-pc_amgx_exact_coarse_solve", "AmgX AMG Exact Coarse Solve", "", amgx->exact_coarse_solve, &amgx->exact_coarse_solve, NULL));
  if (amgx->exact_coarse_solve) amgx->cfg_contents += "exact_coarse_solve=1,";

  amgx->cfg_contents += "solver(amg)=AMG,";

  std::string def_amg_method = map_reverse_lookup(AmgXControlMap::AMGMethods, amgx->amg_method);
  PetscCall(PetscStrncpy(option, def_amg_method.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_amg_method", "AmgX AMG Method", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::AMGMethods.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG Method %s not registered for AmgX.", option);
  amgx->amg_method = AmgXControlMap::AMGMethods.at(option);
  amgx->cfg_contents += "amg:algorithm=" + std::string(option) + ",";

  std::string def_amg_cycle = map_reverse_lookup(AmgXControlMap::AMGCycles, amgx->amg_cycle);
  PetscCall(PetscStrncpy(option, def_amg_cycle.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_amg_cycle", "AmgX AMG Cycle", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::AMGCycles.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG Cycle %s not registered for AmgX.", option);
  amgx->amg_cycle = AmgXControlMap::AMGCycles.at(option);
  amgx->cfg_contents += "amg:cycle=" + std::string(option) + ",";

  std::string def_smoother = map_reverse_lookup(AmgXControlMap::Smoothers, amgx->smoother);
  PetscCall(PetscStrncpy(option, def_smoother.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_smoother", "AmgX Smoother", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::Smoothers.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Smoother %s not registered for AmgX.", option);
  amgx->smoother = AmgXControlMap::Smoothers.at(option);
  amgx->cfg_contents += "amg:smoother(smooth)=" + std::string(option) + ",";

  if (amgx->smoother == AmgXSmoother::JacobiL1 || amgx->smoother == AmgXSmoother::BlockJacobi) {
    PetscCall(PetscOptionsScalar("-pc_amgx_jacobi_relaxation_factor", "AmgX AMG Jacobi Relaxation Factor", "", amgx->jacobi_relaxation_factor, &amgx->jacobi_relaxation_factor, NULL));
    amgx->cfg_contents += "smooth:relaxation_factor=" + std::to_string(amgx->jacobi_relaxation_factor) + ",";
  } else if (amgx->smoother == AmgXSmoother::GS || amgx->smoother == AmgXSmoother::MulticolorGS) {
    PetscCall(PetscOptionsScalar("-pc_amgx_gs_symmetric", "AmgX AMG Gauss Seidel Symmetric", "", amgx->gs_symmetric, &amgx->gs_symmetric, NULL));
    amgx->cfg_contents += "smooth:symmetric_GS=" + std::to_string(amgx->gs_symmetric) + ",";
  }

  std::string def_selector = map_reverse_lookup(AmgXControlMap::Selectors, amgx->selector);
  PetscCall(PetscStrncpy(option, def_selector.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_selector", "AmgX Selector", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);

  /* Validate that the selector is compatible with the chosen AMG method */
  if (amgx->amg_method == AmgXAMGMethod::Classical) {
    PetscCheck(amgx->selector == AmgXSelector::PMIS || amgx->selector == AmgXSelector::HMIS, amgx->comm, PETSC_ERR_PLIB, "Chosen selector is not used for AmgX Classical AMG: selector=%s", option);
    amgx->cfg_contents += "amg:interpolator=D2,";
  } else if (amgx->amg_method == AmgXAMGMethod::Aggregation) {
    PetscCheck(amgx->selector == AmgXSelector::Size2 || amgx->selector == AmgXSelector::Size4 || amgx->selector == AmgXSelector::Size8 || amgx->selector == AmgXSelector::MultiPairwise, amgx->comm, PETSC_ERR_PLIB, "Chosen selector is not used for AmgX Aggregation AMG");
  }
  amgx->selector = AmgXControlMap::Selectors.at(option);
  amgx->cfg_contents += "amg:selector=" + std::string(option) + ",";

  PetscCall(PetscOptionsInt("-pc_amgx_presweeps", "AmgX AMG Presweep Count", "", amgx->presweeps, &amgx->presweeps, NULL));
  amgx->cfg_contents += "amg:presweeps=" + std::to_string(amgx->presweeps) + ",";

  PetscCall(PetscOptionsInt("-pc_amgx_postsweeps", "AmgX AMG Postsweep Count", "", amgx->postsweeps, &amgx->postsweeps, NULL));
  amgx->cfg_contents += "amg:postsweeps=" + std::to_string(amgx->postsweeps) + ",";

  PetscCall(PetscOptionsInt("-pc_amgx_max_levels", "AmgX AMG Max Level Count", "", amgx->max_levels, &amgx->max_levels, NULL));
  amgx->cfg_contents += "amg:max_levels=" + std::to_string(amgx->max_levels) + ",";

  PetscCall(PetscOptionsInt("-pc_amgx_dense_lu_num_rows", "AmgX Dense LU Number of Rows", "", amgx->dense_lu_num_rows, &amgx->dense_lu_num_rows, NULL));
  amgx->cfg_contents += "amg:dense_lu_num_rows=" + std::to_string(amgx->dense_lu_num_rows) + ",";

  PetscCall(PetscOptionsScalar("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));
  amgx->cfg_contents += "amg:strength_threshold=" + std::to_string(amgx->strength_threshold) + ",";

  PetscCall(PetscOptionsInt("-pc_amgx_aggressive_levels", "AmgX AMG Aggressive Coarsening Level Count", "", amgx->aggressive_levels, &amgx->aggressive_levels, NULL));
  if (amgx->aggressive_levels > 0) amgx->cfg_contents += "amg:aggressive_levels=" + std::to_string(amgx->aggressive_levels) + ",";

  std::string def_coarse_solver = map_reverse_lookup(AmgXControlMap::CoarseSolvers, amgx->coarse_solver);
  PetscCall(PetscStrncpy(option, def_coarse_solver.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_coarse_solver", "AmgX CoarseSolver", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::CoarseSolvers.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "CoarseSolver %s not registered for AmgX.", option);
  amgx->coarse_solver = AmgXControlMap::CoarseSolvers.at(option);
  amgx->cfg_contents += "amg:coarse_solver=" + std::string(option) + ",";

  amgx->cfg_contents += "amg:max_iters=1,";

  PetscCall(PetscOptionsBool("-pc_amgx_print_grid_stats", "AmgX Print Grid Stats", "", amgx->print_grid_stats, &amgx->print_grid_stats, NULL));
  if (amgx->print_grid_stats) amgx->cfg_contents += "amg:print_grid_stats=1,";
  amgx->cfg_contents += "amg:monitor_residual=0";

  PetscCall(PetscOptionsBool("-pc_amgx_verbose", "Enable output from AmgX", "", amgx->verbose, &amgx->verbose, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_AMGX(PC pc, PetscViewer viewer)
{
  PC_AMGX  *amgx = (PC_AMGX *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    std::string output_cfg(amgx->cfg_contents);
    std::replace(output_cfg.begin(), output_cfg.end(), ',', '\n');
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n%s\n", output_cfg.c_str()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PCAMGX - Interface to NVIDIA's AmgX algebraic multigrid

  Options Database Keys:
+    -pc_amgx_amg_method (CLASSICAL,AGGREGATION)                       - set the AMG algorithm to use
.    -pc_amgx_amg_cycle (V,W,F,CG)                                     - set the AMG cycle type
.    -pc_amgx_jacobi_relaxation_factor                                 - set the relaxation factor for Jacobi smoothing
.    -pc_amgx_gs_symmetric                                             - enforce symmetric Gauss-Seidel smoothing (only applies if GS smoothing is selected)
.    -pc_amgx_selector (SIZE_2|SIZE_4|SIZE_8|MULTI_PAIRWISE|PMIS|HMIS) - set the AMG coarse selector
.    -pc_amgx_presweeps                                                - set the number of AMG pre-sweeps
.    -pc_amgx_postsweeps                                               - set the number of AMG post-sweeps
.    -pc_amgx_max_levels                                               - set the maximum number of levels in the AMG level hierarchy
.    -pc_amgx_strength_threshold                                       - set the strength threshold for the AMG coarsening
.    -pc_amgx_aggressive_levels                                        - set the number of levels (from the finest) that should apply aggressive coarsening
.    -pc_amgx_coarse_solver (DENSE_LU_SOLVER,NOSOLVER)                 - set the coarse solve
.    -pc_amgx_print_grid_stats                                         - output the AMG grid hierarchy to `stdout`
-    -pc_amgx_verbose                                                  - enable AmgX verbose output
-    -pc_amgx_smoother (PCG|PCGF|PBICGSTAB|GMRES|FGMRES|JACOBI_L1|BLOCK_JACOBI|GS|MULTICOLOR_GS|MULTICOLOR_ILU|MULTICOLOR_DILU|CHEBYSHEV_POLY|NOSOLVER) - set the AMG pre/post smoother

   Level: intermediate

   Note:
   Implementation will accept host or device pointers, but good performance will require that the `KSP` is also GPU accelerated so that data is not frequently transferred between host and device.

.seealso: [](ch_ksp), `PCGAMG`, `PCHYPRE`, `PCMG`, `PCAmgXGetResources()`, `PCCreate()`, `PCSetType()`, `PCType`, `PC`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_AMGX(PC pc)
{
  PC_AMGX *amgx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&amgx));
  pc->ops->apply          = PCApply_AMGX;
  pc->ops->setfromoptions = PCSetFromOptions_AMGX;
  pc->ops->setup          = PCSetUp_AMGX;
  pc->ops->view           = PCView_AMGX;
  pc->ops->destroy        = PCDestroy_AMGX;
  pc->ops->reset          = PCReset_AMGX;
  pc->data                = (void *)amgx;

  amgx->selector                 = AmgXSelector::PMIS;
  amgx->smoother                 = AmgXSmoother::BlockJacobi;
  amgx->amg_method               = AmgXAMGMethod::Classical;
  amgx->coarse_solver            = AmgXCoarseSolver::DenseLU;
  amgx->amg_cycle                = AmgXAMGCycle::V;
  amgx->exact_coarse_solve       = PETSC_TRUE;
  amgx->presweeps                = 1;
  amgx->postsweeps               = 1;
  amgx->max_levels               = 100;
  amgx->strength_threshold       = 0.5;
  amgx->aggressive_levels        = 0;
  amgx->dense_lu_num_rows        = 1;
  amgx->jacobi_relaxation_factor = 0.9;
  amgx->gs_symmetric             = PETSC_FALSE;
  amgx->print_grid_stats         = PETSC_FALSE;
  amgx->verbose                  = PETSC_FALSE;
  amgx->rsrc_init                = PETSC_FALSE;
  amgx->solve_state_init         = PETSC_FALSE;

  s_count++;

  PetscCallCUDA(cudaGetDevice(&amgx->devID));
  if (s_count == 1) {
    PetscCallAmgX(AMGX_initialize());
    PetscCallAmgX(AMGX_initialize_plugins());
    PetscCallAmgX(AMGX_register_print_callback(&print_callback));
    PetscCallAmgX(AMGX_install_signal_handler());
  }
  /* This communicator is not yet known to this system, so we duplicate it and make an internal communicator */
  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)pc), &amgx->comm));
  PetscCallMPI(MPI_Comm_size(amgx->comm, &amgx->nranks));
  PetscCallMPI(MPI_Comm_rank(amgx->comm, &amgx->rank));

  PetscCall(amgx_output_messages(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCAmgXGetResources - get AmgX's internal resource object

  Not Collective, No Fortran Support

  Input Parameter:
. pc - the PC

  Output Parameter:
. rsrc_out - pointer to the AmgX resource object

  Level: advanced

.seealso: [](ch_ksp), `PCAMGX`, `PC`, `PCGAMG`
@*/
PETSC_EXTERN PetscErrorCode PCAmgXGetResources(PC pc, void *rsrc_out)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (!amgx->rsrc_init) {
    PetscCallAmgX(AMGX_config_create(&amgx->cfg, amgx->cfg_contents.c_str()));
    PetscCallAmgX(AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &amgx->devID));
    amgx->rsrc_init = PETSC_TRUE;
  }
  *(AMGX_resources_handle *)rsrc_out = amgx->rsrc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
