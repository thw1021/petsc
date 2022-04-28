/*  --------------------------------------------------------------------

     This file implements a AMGx preconditioner in PETSc as part of PC.

    -------------------------------------------------------------------- */

/*
   Include files needed for the AMGX preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscdevice.h>
#include <amgx_c.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <map>
#include "cuda_runtime.h"

enum class AmgXSmoother { PCG, PCGF, PBiCGStab, GMRES, FGMRES, JacobiL1,
  BlockJacobi, GS, MulticolorGS, MulticolorILU, MulticolorDILU, ChebyshevPoly, NoSolver };
enum class AmgXAMGMethod { Classical, Aggregation };
enum class AmgXSelector { Size2, Size4, Size8, MultiPairwise, PMIS, HMIS };
enum class AmgXCoarseSolver { DenseLU, NoSolver };
enum class AmgXAMGCycle { V, W, F, CG, CGF };

struct AmgXControlMap
{
  static const std::map<std::string, AmgXAMGMethod> AMGMethods;
  static const std::map<std::string, AmgXSmoother> Smoothers;
  static const std::map<std::string, AmgXSelector> Selectors;
  static const std::map<std::string, AmgXCoarseSolver> CoarseSolvers;
  static const std::map<std::string, AmgXAMGCycle> AMGCycles;
};

const std::map<std::string, AmgXAMGMethod> AmgXControlMap::AMGMethods =
{
  { "CLASSICAL", AmgXAMGMethod::Classical },
  { "AGGREGATION", AmgXAMGMethod::Aggregation }
};

const std::map<std::string, AmgXSmoother> AmgXControlMap::Smoothers =
{
  { "PCG", AmgXSmoother::PCG },
  { "PCGF", AmgXSmoother::PCGF },
  { "PBICGSTAB", AmgXSmoother::PBiCGStab },
  { "GMRES", AmgXSmoother::GMRES },
  { "FGMRES", AmgXSmoother::FGMRES },
  { "JACOBI_L1", AmgXSmoother::JacobiL1 },
  { "BLOCK_JACOBI", AmgXSmoother::BlockJacobi },
  { "GS", AmgXSmoother::GS },
  { "MULTICOLOR_GS", AmgXSmoother::MulticolorGS },
  { "MULTICOLOR_ILU", AmgXSmoother::MulticolorILU },
  { "MULTICOLOR_DILU", AmgXSmoother::MulticolorDILU },
  { "CHEBYSHEV_POLY", AmgXSmoother::ChebyshevPoly },
  { "NOSOLVER", AmgXSmoother::NoSolver }
};

const std::map<std::string, AmgXSelector> AmgXControlMap::Selectors =
{
  { "SIZE_2", AmgXSelector::Size2 },
  { "SIZE_4", AmgXSelector::Size4 },
  { "SIZE_8", AmgXSelector::Size8 },
  { "MULTI_PAIRWISE", AmgXSelector::MultiPairwise },
  { "PMIS", AmgXSelector::PMIS },
  { "HMIS", AmgXSelector::HMIS }
};

const std::map<std::string, AmgXCoarseSolver> AmgXControlMap::CoarseSolvers =
{
  { "DENSE_LU_SOLVER", AmgXCoarseSolver::DenseLU },
  { "NOSOLVER", AmgXCoarseSolver::NoSolver }
};

const std::map<std::string, AmgXAMGCycle> AmgXControlMap::AMGCycles =
{
  { "V", AmgXAMGCycle::V },
  { "W", AmgXAMGCycle::W },
  { "F", AmgXAMGCycle::F },
  { "CG", AmgXAMGCycle::CG },
  { "CGF", AmgXAMGCycle::CGF }
};

/*
   Private context (data structure) for the AMGX preconditioner.
*/
struct PC_AMGX {
  AMGX_solver_handle solver;
  AMGX_config_handle cfg;
  AMGX_resources_handle rsrc;
  bool rsrc_init = false;

  AMGX_matrix_handle A;
  AMGX_vector_handle sol;
  AMGX_vector_handle rhs;

  MPI_Comm comm;
  int rank = 0;
  int nranks = 0;
  int devID = 0;

  void *lib_handle = 0;
  std::string cfg_contents;

  // Cached state for re-setup
  PetscInt nnz;
  PetscInt nLocalRows;
  PetscInt nGlobalRows;
  PetscInt bSize;
  Mat localA;
  const PetscScalar *values;

  // AMG Control parameters
  AmgXSmoother smoother;
  AmgXAMGMethod amg_method;
  AmgXSelector selector;
  AmgXCoarseSolver coarse_solver;
  AmgXAMGCycle amg_cycle;
  PetscInt presweeps;
  PetscInt postsweeps;
  PetscInt max_levels;
  PetscInt aggressive_levels;
  PetscScalar strength_threshold;
  PetscBool print_grid_stats;

  // Smoother control parameters
  PetscScalar jacobi_relaxation_factor;
  PetscScalar gs_symmetric;
};

static PetscInt s_count = 0;

/* print callback (could be customized) */
static void print_callback(const char *msg, int length)
{
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (rank == 0) {
    PetscPrintf(PETSC_COMM_SELF,"%s", msg);
  }
}

// XXX Presumably PETSc has some routines that can be used instead here?
PetscErrorCode print_error(char const* file, int const line, cudaError_t error)
{
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SIG, "Error: %s:%d, code:%d, name: %s, reason: %s",file, line, error, cudaGetErrorName(error), cudaGetErrorString(error));
}

/** \brief A macro to check the returned CUDA error code.
 *
 * \param call [in] Function call to CUDA API.
 */
#define CHECK(call)                             \
{                                               \
  const cudaError_t error = call;               \
  if (error != cudaSuccess) {                   \
    print_error(__FILE__, __LINE__, error);     \
  }                                             \
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_AMGX - Prepares for the use of the AMGX preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;
  Mat Pmat = pc->pmat;

  PetscFunctionBegin;

  if (!pc->setupcalled) {
    // Initialise resources and matrices
    if (!amgx->rsrc_init) {
      // Read configuration file and set exception handling
      AMGX_SAFE_CALL(AMGX_config_create(&amgx->cfg, amgx->cfg_contents.c_str()));

      /* switch on internal error handling (no need to use AMGX_SAFE_CALL after this point) */
      AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "exception_handling=1"));

      AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &amgx->devID);

      amgx->rsrc_init = true;
    }

    AMGX_matrix_create(&amgx->A, amgx->rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx->sol, amgx->rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx->rhs, amgx->rsrc, AMGX_mode_dDDI);
    AMGX_solver_create(&amgx->solver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg);

    // BUG If PetscInt is 64-bit and int is 32-bit this will lead to a
    // bug, as passed through to AmgX as PetscInt, but expects int

    // At the present time, an AmgX matrix is a sequential matrix
    // Non-sequential/MPI matrices must be adapted to extract the local matrix
    if (amgx->nranks > 1) {
      PetscCall(MatMPIAIJGetLocalMat(Pmat, MAT_INITIAL_MATRIX, &amgx->localA));
    } else {
      amgx->localA = Pmat;
    }

    // Extract the CSR data
    PetscBool done;
    const PetscInt *colIndices;
    const PetscInt *rowOffsets;
    PetscCall(MatGetRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &amgx->nLocalRows, &rowOffsets, &colIndices, &done));
    PetscCheck(done, amgx->comm, PETSC_ERR_PLIB, "MatGetRowIJ was not successful");

    PetscCheck(amgx->nLocalRows < std::numeric_limits<int>::max(), PETSC_COMM_SELF,PETSC_ERR_PLIB, "AmgX restricted to int local rows but nLocalRows = %" PetscInt_FMT " > max<int>", amgx->nLocalRows);

    PetscCall(MatSeqAIJGetArrayRead(amgx->localA, &amgx->values));

    amgx->nnz = rowOffsets[amgx->nLocalRows];

    PetscCheck(amgx->nnz < std::numeric_limits<int>::max(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support for 64-bit integer nnz not yet implemented, nnz = %" PetscInt_FMT ".", amgx->nnz);

    // Allocate space for some partition offsets
    std::vector<PetscInt> partitionOffsets(amgx->nranks + 1);

    // Fetch the number of local rows per rank
    partitionOffsets[0] = 0; /* could use PetscLayoutGetRanges */
    PetscCall(MPIU_Allgather(&amgx->nLocalRows, 1, MPIU_INT, partitionOffsets.data()+1, 1, MPIU_INT, amgx->comm));

    // Prefix sum to get offsets
    std::partial_sum(partitionOffsets.begin(),partitionOffsets.end(),partitionOffsets.begin());

    // Fetch the number of global rows
    amgx->nGlobalRows = partitionOffsets[amgx->nranks];

    PetscCall(MatGetBlockSize(Pmat, &amgx->bSize));

    // XXX Currently constrained to 32-bit indices, to be changed in the future
    // Create the distribution and upload the matrix data
    AMGX_distribution_handle dist;
    AMGX_distribution_create(&dist, amgx->cfg);
    AMGX_distribution_set_32bit_colindices(dist, true);
    AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partitionOffsets.data());

    AMGX_matrix_upload_distributed(amgx->A, amgx->nGlobalRows, (int)amgx->nLocalRows, (int)amgx->nnz, amgx->bSize, amgx->bSize, rowOffsets, colIndices, amgx->values, NULL, dist);

    PetscCallMPI(MPI_Barrier(amgx->comm));

    AMGX_solver_setup(amgx->solver, amgx->A);
    AMGX_vector_bind(amgx->sol, amgx->A);
    AMGX_vector_bind(amgx->rhs, amgx->A);

    PetscInt nlr = 0;
    PetscCall(MatRestoreRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &nlr, &rowOffsets, &colIndices, &done));

  } else {
    // The fast path after the initial setup phase
    AMGX_matrix_replace_coefficients(amgx->A, amgx->nLocalRows, amgx->nnz, amgx->values, NULL);

    AMGX_solver_resetup(amgx->solver, amgx->A);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_AMGX - Applies the AMGX preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  b - rhs vector

   Output Parameter:
.  x - solution vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_AMGX(PC pc, Vec b, Vec x)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;

  PetscScalar *x_;
  const PetscScalar *b_;

  PetscBool is_dev_ptrs;
  PetscCall(PetscObjectTypeCompare((PetscObject)x, VECSEQCUDA, &is_dev_ptrs));

  if (is_dev_ptrs) {
    PetscCall(VecCUDAGetArrayWrite(x, &x_));
    PetscCall(VecCUDAGetArrayRead(b, &b_));
  } else {
    PetscCall(VecGetArrayWrite(x, &x_));
    PetscCall(VecGetArrayRead(b, &b_));
  }

  AMGX_vector_upload(amgx->sol, amgx->nLocalRows, 1, x_);
  AMGX_vector_upload(amgx->rhs, amgx->nLocalRows, 1, b_);

  PetscCallMPI(MPI_Barrier(amgx->comm));

  AMGX_solver_solve_with_0_initial_guess(amgx->solver, amgx->rhs, amgx->sol);

  AMGX_SOLVE_STATUS status;
  AMGX_solver_get_status(amgx->solver, &status);

  // needs some thought
  if (status == AMGX_SOLVE_FAILED) PetscCall(PCSetErrorIfFailure(pc, PETSC_TRUE));
  PetscCheck(status != AMGX_SOLVE_FAILED, amgx->comm, PETSC_ERR_CONV_FAILED, "AmgX solver failed to solve the system! The error code is %d.", status);

  AMGX_vector_download(amgx->sol, x_);

  if (is_dev_ptrs) {
    PetscCall(VecCUDARestoreArrayWrite(x, &x_));
    PetscCall(VecCUDARestoreArrayRead(b, &b_));
  } else {
    PetscCall(VecRestoreArrayWrite(x, &x_));
    PetscCall(VecRestoreArrayRead(b, &b_));
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode PCReset_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  if (pc->setupcalled) {
    PetscCall(MatSeqAIJRestoreArrayRead(amgx->localA, &amgx->values));
  }

  PetscFunctionBegin;
  AMGX_solver_destroy(amgx->solver);
  AMGX_matrix_destroy(amgx->A);
  AMGX_vector_destroy(amgx->sol);
  AMGX_vector_destroy(amgx->rhs);
  PetscFunctionReturn(0);
}

/*
   PCDestroy_AMGX - Destroys the private context for the AMGX preconditioner
   that was created with PCCreate_AMGX().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;

  /* decrease the number of instances, only the last instance need to destroy resource and finalizing AmgX */
  if (s_count == 1) {
    /* can put this in a PCAMGXInitializePackage method */
    PetscCheck(amgx->rsrc != nullptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "s_rsrc == NULL");

    AMGX_resources_destroy(amgx->rsrc);

    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(amgx->cfg));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());

    PetscCallMPI(MPI_Comm_free(&amgx->comm));
#ifdef AMGX_DYNAMIC_LOADING
    amgx_libclose(amgx->lib_handle);
#endif
  } else {
    AMGX_SAFE_CALL(AMGX_config_destroy(amgx->cfg));
  }
  s_count -= 1;
  PetscCall(PetscFree(amgx));

  PetscFunctionReturn(0);
}

template <class T>
std::string map_reverse_lookup(const std::map<std::string, T>& map, const T& key)
{
  for (auto const& m : map) {
    if (m.second == key) {
      return m.first;
    }
  }

  return "";
}

static PetscErrorCode PCSetFromOptions_AMGX(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;

  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  // Set the defaults
  amgx->selector = AmgXSelector::PMIS;
  amgx->smoother = AmgXSmoother::BlockJacobi;
  amgx->amg_method = AmgXAMGMethod::Classical;
  amgx->coarse_solver = AmgXCoarseSolver::DenseLU;
  amgx->amg_cycle = AmgXAMGCycle::V;
  amgx->presweeps = 1;
  amgx->postsweeps = 1;
  amgx->max_levels = 100;
  amgx->strength_threshold = 0.5;
  amgx->aggressive_levels = 0;
  amgx->jacobi_relaxation_factor = 0.9;
  amgx->gs_symmetric = PETSC_FALSE;
  amgx->print_grid_stats = PETSC_FALSE;

  constexpr int MAX_PARAM_LEN = 128;
  char option[MAX_PARAM_LEN];

  PetscOptionsHeadBegin(PetscOptionsObject, "AmgX options");

  amgx->cfg_contents = "config_version=2,";
  amgx->cfg_contents += "determinism_flag=1,";
  amgx->cfg_contents += "solver(amg)=AMG,";

  // Set method
  std::string def_amg_method = map_reverse_lookup(AmgXControlMap::AMGMethods, amgx->amg_method);
  PetscCall(PetscStrcpy(option, def_amg_method.c_str()));
  PetscCall(PetscOptionsString("-pc_amgx_amg_method", "AmgX AMG Method", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::AMGMethods.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG Method %s not registered for AmgX.", option);
  amgx->amg_method = AmgXControlMap::AMGMethods.at(option);
  amgx->cfg_contents += "amg:algorithm=" + std::string(option) + ",";

  // Set cycle
  std::string def_amg_cycle = map_reverse_lookup(AmgXControlMap::AMGCycles, amgx->amg_cycle);
  PetscCall(PetscStrcpy(option, def_amg_cycle.c_str()));
  PetscCall(PetscOptionsString("-pc_amgx_amg_cycle", "AmgX AMG Cycle", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::AMGCycles.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG Cycle %s not registered for AmgX.", option);
  amgx->amg_cycle = AmgXControlMap::AMGCycles.at(option);
  amgx->cfg_contents += "amg:cycle=" + std::string(option) + ",";

  // Set smoother
  std::string def_smoother = map_reverse_lookup(AmgXControlMap::Smoothers, amgx->smoother);
  PetscCall(PetscStrcpy(option, def_smoother.c_str()));
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

  // Set selector
  std::string def_selector = map_reverse_lookup(AmgXControlMap::Selectors, amgx->selector);
  PetscCall(PetscStrcpy(option, def_selector.c_str()));
  PetscCall(PetscOptionsString("-pc_amgx_selector", "AmgX Selector", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);

  // Double check that the user has selected an appropriate selector for the AMG method
  if (amgx->amg_method == AmgXAMGMethod::Classical) {
    PetscCheck(amgx->selector == AmgXSelector::PMIS || amgx->selector == AmgXSelector::HMIS, amgx->comm, PETSC_ERR_PLIB, "Chosen selector is not used for AmgX Classical AMG: selector=%s", option);

    amgx->cfg_contents += "amg:interpolator=D2,";
  } else if (amgx->amg_method == AmgXAMGMethod::Aggregation) {
    PetscCheck(amgx->selector == AmgXSelector::Size2 || amgx->selector == AmgXSelector::Size4 || amgx->selector == AmgXSelector::Size8 || amgx->selector == AmgXSelector::MultiPairwise, amgx->comm, PETSC_ERR_PLIB, "Chosen selector is not used for AmgX Aggregation AMG");
  }
  amgx->selector = AmgXControlMap::Selectors.at(option);
  amgx->cfg_contents += "amg:selector=" + std::string(option) + ",";

  // Set presweeps
  PetscCall(PetscOptionsInt("-pc_amgx_presweeps", "AmgX AMG Presweep Count", "", amgx->presweeps, &amgx->presweeps, NULL));
  amgx->cfg_contents += "amg:presweeps=" + std::to_string(amgx->presweeps) + ",";

  // Set postsweeps
  PetscCall(PetscOptionsInt("-pc_amgx_postsweeps", "AmgX AMG Postsweep Count", "", amgx->postsweeps, &amgx->postsweeps, NULL));
  amgx->cfg_contents += "amg:postsweeps=" + std::to_string(amgx->postsweeps) + ",";

  // Set max levels
  PetscCall(PetscOptionsInt("-pc_amgx_max_levels", "AmgX AMG Max Level Count", "", amgx->max_levels, &amgx->max_levels, NULL));
  amgx->cfg_contents += "amg:max_levels=100,";

  // Set strength threshold
  PetscCall(PetscOptionsScalar("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));
  amgx->cfg_contents += "amg:strength_threshold=" + std::to_string(amgx->strength_threshold) + ",";

  // Set aggressive_levels
  PetscCall(PetscOptionsInt("-pc_amgx_aggressive_levels", "AmgX AMG Presweep Count", "", amgx->aggressive_levels, &amgx->aggressive_levels, NULL));

  if (amgx->aggressive_levels > 0) {
    amgx->cfg_contents += "amg:aggressive_levels=" + std::to_string(amgx->aggressive_levels) + ",";
  }

  // Set coarse solver
  std::string def_coarse_solver = map_reverse_lookup(AmgXControlMap::CoarseSolvers, amgx->coarse_solver);
  PetscCall(PetscStrcpy(option, def_coarse_solver.c_str()));
  PetscCall(PetscOptionsString("-pc_amgx_coarse_solver", "AmgX CoarseSolver", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::CoarseSolvers.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "CoarseSolver %s not registered for AmgX.", option);
  amgx->coarse_solver = AmgXControlMap::CoarseSolvers.at(option);
  amgx->cfg_contents += "amg:coarse_solver=" + std::string(option) + ",";

  // Set max iterations
  amgx->cfg_contents += "amg:max_iters=1,";

  // Set output control parameters
  PetscCall(PetscOptionsBool("-pc_amgx_print_grid_stats", "AmgX Print Grid Stats", "", amgx->print_grid_stats, &amgx->print_grid_stats, NULL));

  if (amgx->print_grid_stats) {
    amgx->cfg_contents += "amg:print_grid_stats=1,";
  }

  amgx->cfg_contents += "amg:monitor_residual=0";

  PetscOptionsHeadEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_AMGX(PC pc, PetscViewer viewer)
{
  PetscFunctionBegin;

  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscBool iascii;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    std::string output_cfg(amgx->cfg_contents);
    std::replace(output_cfg.begin(), output_cfg.end(), ',', '\n');
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n%s\n", output_cfg.c_str()));
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_AMGX - Creates a AMGX preconditioner context, PC_AMGX,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCAMGX - AMGX (i.e. diagonal scaling preconditioning)

   Options Database Key:
+    -pc_amgx_type <diagonal,rowmax,rowsum> - approach for forming the preconditioner
-    -pc_amgx_fixdiag - fix for zero diagonal terms

   Level: beginner

  Notes:
    By using KSPSetPCSide(ksp,PC_SYMMETRIC) or -ksp_pc_side symmetric
         can scale each side of the matrix by the square root of the diagonal entries.

         Zero entries along the diagonal are replaced with the value 1.0

         See PCPBAMGX for a point-block AMGX preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCAMGXSetType(), PCAMGXSetUseAbs(), PCAMGXGetUseAbs(),
           PCPBAMGX
M*/

PETSC_EXTERN PetscErrorCode PCCreate_AMGX(PC pc)
{
  PC_AMGX *amgx;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc, &amgx));
  pc->ops->apply = PCApply_AMGX;
  pc->ops->setfromoptions = PCSetFromOptions_AMGX;
  pc->ops->setup = PCSetUp_AMGX;
  pc->ops->view = PCView_AMGX;
  pc->ops->destroy = PCDestroy_AMGX;
  pc->ops->reset = PCReset_AMGX;
  pc->data = (void *)amgx;

  s_count++;

  PetscCallCUDA(cudaGetDevice(&amgx->devID));

  if (s_count == 1) {

    /* can put this in a PCAMGXFinalizePackage method */
    /* load the library (if it was dynamically loaded) */
#ifdef AMGX_DYNAMIC_LOADING
    amgx->lib_handle = NULL;
#ifdef _WIN32
    amgx->lib_handle = amgx_libopen("amgxsh.dll");
#else
    amgx->lib_handle = amgx_libopen("libamgxsh.so");
#endif
    if (amgx->lib_handle == NULL) {
          errAndExit("ERROR: can not load the library");
    }
    //load all the routines
    if (amgx_liblink_all(amgx->lib_handle) == 0) {
      amgx_libclose(amgx->lib_handle);
      errAndExit("ERROR: corrupted library loaded\n");
    }
#endif
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
  }
  /* This communicator is not yet known to this system, so we duplicate it and make an internal communicator */
  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)pc), &amgx->comm));

  PetscCallMPI(MPI_Comm_size(amgx->comm, &amgx->nranks));
  PetscCallMPI(MPI_Comm_rank(amgx->comm, &amgx->rank));

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PCAmgXGetResources(PC pc, void* rsrc_out)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (!amgx->rsrc_init) {
    // Read configuration file and set exception handling
    AMGX_SAFE_CALL(AMGX_config_create(&amgx->cfg, amgx->cfg_contents.c_str()));

    /* switch on internal error handling (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "exception_handling=1"));

    AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &amgx->devID);

    amgx->rsrc_init = true;
  }

  *static_cast<AMGX_resources_handle*>(rsrc_out) = amgx->rsrc;
  PetscFunctionReturn(0);
}

