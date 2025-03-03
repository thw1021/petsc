static char help[] = "Test batched TS solves\n";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscksp.h>
#include <petscts.h>
#include <petscdmswarm.h>
#include <petsc/private/petscfeimpl.h>//For computation of the field gradient.
#include <petscdt.h>
#include <petsc/private/tsimpl.h>

// Hip inclusions should be wrapped up into petsc device calls in the TS
#include <petscdevice_hip.h>
#include "../impls/device/tsdevice.hip.cpp"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <petsctsdevice.hpp>

/*
  Struct to store problem parameters and pass into petsc objects
*/
typedef struct {
    PetscInt n_pdes;
    PetscInt dim; // Remove when the dm is used and just get it from that.
    DM       dm;
} AppCtx;

/*
  Parse user options
*/
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{

  PetscFunctionBeginUser;
  options->n_pdes = 1;
  options->dim = 2;

  PetscOptionsBegin(comm, "", "Batched TS example options", "");
  PetscCall(PetscOptionsInt("-n_pdes", "Number of PDEs to solve", "ex16.hip.cc", options->n_pdes, &options->n_pdes, NULL));
  PetscCall(PetscOptionsInt("-dim", "Dimension of the system.", "ex16.hip.cc", options->dim, &options->dim, NULL));//TODO: use the DM for this.
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  user->dm = *dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
// Can move out into kernel function but I don't see any real speedup there
static PetscErrorCode ComputeIC(TS ts, Vec u){
    PetscScalar *ic;
    PetscRandom rnd;
    AppCtx *user;
    PetscFunctionBegin;
    PetscCall(TSGetApplicationContext(ts, (void *)&user));
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
    PetscCall(PetscRandomSetType(rnd, PETSCRAND48));
    PetscCall(PetscRandomSetInterval(rnd, 10., 20.));
    PetscCall(VecGetArray(u, &ic));
    for (PetscInt i = 0; i < user->n_pdes; ++i){
        PetscReal vel;
        ic[i*2*user->dim + 0] = 0.;// Two elements per dimension: x, y, vx, vy...
        ic[i*2*user->dim + 1] = 0.;
        PetscCall(PetscRandomGetValueReal(rnd, &vel));
        ic[i*2*user->dim + 2] = vel;
        PetscCall(PetscRandomGetValueReal(rnd, &vel));
        ic[i*2*user->dim + 3] = vel;
    }
    PetscCall(VecRestoreArray(u, &ic));
    PetscCall(PetscRandomDestroy(&rnd));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Evolve the trajectory of a group of balls given a set of initial conditions
  and g = 9.8m/s^2.
*/
struct func{
    const PetscReal g = 9.8;
    __device__ void operator()(TSDevice_Euler ts, PetscReal t, PetscScalar *u, PetscScalar *f, void *ctx){
    int i = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
    int elements = ts.elements;
    printf("Thread %d rhs function for %d elements.\n", i, elements);
    f[i*elements + 0] = u[i*elements + 2];// update on x using vx
    f[i*elements + 1] = u[i*elements + 3];// update on y using vy
    f[i*elements + 2] = 0.;// No drag
    f[i*elements + 3] = -9.8;// gravity
  }
  // Very simple function to check if the solution has gone below the x axis for the TSDevice_Event
  __device__ void operator()(PetscReal update[], PetscReal *indicator){
      *indicator = update[1];
  }
  // Very simple post event function.
  __device__ void operator()(TSDevice_Euler ts, PetscReal dt, PetscReal *solution){
      solution[3] = -solution[3]; // Reverse the velocity in the y direction
  }
};
// rocprofv2 -d output --hip-trace ./ex5 -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0.0,0.0,0.0 -dm_plex_box_upper 0.1,25.,10. -dm_plex_box_faces 1,1,16 -ts_type euler -petscfe_default_quadrature_order 3 -petscspace_degree 3 -ts_dt .01 -ts_max_steps 100 -post_step_view -dm_vec_type hip -vec_type hip
int main(int argc, char *argv[])
{
  TS       ts;
  Vec      sol;
  MPI_Comm comm;
  AppCtx   user;

  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(VecCreate(comm, &sol));
  PetscCall(VecSetSizes(sol, user.n_pdes*2*user.dim, PETSC_DECIDE));//Space for x, y, vx, vy...
  PetscCall(VecSetFromOptions(sol));
  //PetscCall(VecSet(sol, 10.0));// 2D, set x1,y1,vx1,vy1, x2, y2.... etc.
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetFromOptions(ts));
  //PetscCall(TSDeviceSetRHSFunction(ts, NULL, RHSFunctionBall, NULL));
  PetscCall(TSSetApplicationContext(ts, (void *)&user));
  PetscCall(TSDeviceSetNumEquations(ts, user.n_pdes));// number of pdes in the system, ie, there will be 2 worker threads.
  PetscCall(TSDeviceSetNumElements(ts, 2*user.dim));// Full configuration space.
  PetscCall(TSSetComputeInitialCondition(ts, ComputeIC));
  PetscCall(TSSetSolution(ts, sol));
  PetscCall(TSComputeInitialCondition(ts, sol));
  func rhsfunc;
  PetscCall(TSSolve_Device(ts, rhsfunc));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&sol));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex double

  test:
    suffix: ball_launch
    args: -ts_type device -ts_batch_ts_type euler -ts_max_steps 5 -vec_type hip -n_pdes 2
    filter: grep -v marker | grep -v atomic | grep -v usage
TEST*/
