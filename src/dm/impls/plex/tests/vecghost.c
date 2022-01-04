static char help[] = "Tests DMPlexVecGhost() for a simple finite element mesh.\n\n";

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscsf.h>

enum {STAGE_LOAD, STAGE_DISTRIBUTE, STAGE_REFINE, STAGE_OVERLAP};

typedef struct {
  PetscLogEvent createMeshEvent;
  PetscLogStage stages[4];
  /* Domain and mesh definition */
  PetscInt      dim;                             /* The topological mesh dimension */
  PetscInt      overlap;                         /* The cell overlap to use during partitioning */
  PetscBool     final_ref;                       /* Run refinement at the end */
  PetscBool     final_diagnostics;               /* Run diagnostics on the final mesh */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim               = 2;
  options->overlap           = 0;
  options->final_ref         = PETSC_FALSE;
  options->final_diagnostics = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-overlap", "The cell overlap for partitioning", "ex1.c", options->overlap, &options->overlap, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-final_ref", "Run uniform refinement on the final mesh", "ex1.c", options->final_ref, &options->final_ref, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-final_diagnostics", "Run diagnostics on the final mesh", "ex1.c", options->final_diagnostics, &options->final_diagnostics, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscLogEventRegister("CreateMesh",DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshLoad",&options->stages[STAGE_LOAD]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshDistribute",&options->stages[STAGE_DISTRIBUTE]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshRefine",&options->stages[STAGE_REFINE]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshOverlap",&options->stages[STAGE_OVERLAP]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim           = user->dim;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;
  DM             distributedMesh;
  PetscSF        sf;
  PetscInt       nranks;
  Vec            v;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = PetscLogStagePush(user->stages[STAGE_LOAD]);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);

  ierr = DMViewFromOptions(*dm,NULL,"-init_dm_view");CHKERRQ(ierr);
  ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);

  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stages[STAGE_DISTRIBUTE]);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_pre_dist_view");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, "dist_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, NULL);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-distributed_dm_view");CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stages[STAGE_REFINE]);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, "ref_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, NULL);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = DMViewFromOptions(*dm, NULL, "-dm_pre_redist_view");CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMGetPointSF(distributedMesh, &sf);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = DMGetNeighbors(distributedMesh, &nranks, NULL);CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &nranks, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)*dm));CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)*dm)), "Minimum number of neighbors: %D\n", nranks);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distributedMesh;
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_post_redist_view");CHKERRQ(ierr);

  if (user->final_ref) {
    DM refinedMesh = NULL;

    ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
  }

  ierr = DMPlexSetUseVecGhostPermutation(*dm);CHKERRQ(ierr);
  {
    PetscFE fe;

    /* piecewise linear finite elements */
    ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_TRUE, NULL,PETSC_DETERMINE, &fe);CHKERRQ(ierr);
    ierr = DMAddField(*dm, NULL, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMCreateDS(*dm);CHKERRQ(ierr);

    ierr = DMPlexCreateGhostVector(*dm,&v);CHKERRQ(ierr);
    {
      PetscInt    i,rstart,rend;
      Vec         lv;
      PetscScalar value;
      PetscViewer viewer;

      ierr = VecGetOwnershipRange(v,&rstart,&rend);CHKERRQ(ierr);
      for (i=rstart; i<rend; i++) {
        value = i+1;
        ierr  = VecSetValues(v,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
      ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGhostGetLocalForm(v,&lv);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
      ierr = VecView(lv,viewer);CHKERRQ(ierr);
      ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
      ierr = VecGhostRestoreLocalForm(v,&lv);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Generated Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  if (user->final_diagnostics) {
    DMPlexInterpolatedFlag interpolated;
    PetscInt  dim, depth;

    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(*dm, &depth);CHKERRQ(ierr);
    ierr = DMPlexIsInterpolatedCollective(*dm, &interpolated);CHKERRQ(ierr);

    ierr = DMPlexCheckSymmetry(*dm);CHKERRQ(ierr);
    if (interpolated == DMPLEX_INTERPOLATED_FULL) {
      ierr = DMPlexCheckFaces(*dm, 0);CHKERRQ(ierr);
    }
    ierr = DMPlexCheckSkeleton(*dm, 0);CHKERRQ(ierr);
    //    ierr = DMPlexCheckGeometry(*dm);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    nsize: 3
    args: -dm_view ::ascii_info_detail -petscpartitioner_type simple

  test:
    suffix: 1
    requires: triangle
    nsize: 2
    args: -dm_view ::ascii_info_detail -petscpartitioner_type simple

TEST*/
