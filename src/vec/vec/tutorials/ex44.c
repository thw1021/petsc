
static char help[] = "Test VecConcatenate both in serial and parallel.\n";

#include <petscvec.h>

int main(int argc,char **args)
{
  Vec                u, v, w, u_test, v_test, w_test;
  IS                 u_is, v_is;
  VecScatter         w_to_u, w_to_v;
  PetscInt           i, *u_idx, *v_idx, *w_idx;
  PetscScalar        *u_val, *v_val, *w_val;
  PetscBool          u_equal, v_equal, w_equal;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, &u);CHKERRQ(ierr);
  ierr = VecSetSizes(u, PETSC_DECIDE, 4);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecSetUp(u);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &v);CHKERRQ(ierr);
  ierr = VecSetSizes(v, PETSC_DECIDE, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = VecSetUp(v);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &w);CHKERRQ(ierr);
  ierr = VecSetSizes(w, PETSC_DECIDE, 6);CHKERRQ(ierr);
  ierr = VecSetFromOptions(w);CHKERRQ(ierr);
  ierr = VecSetUp(w);CHKERRQ(ierr);

  ierr = PetscMalloc1(4, &u_idx);
  ierr = PetscMalloc1(4, &u_val);
  for (i=0; i<4; i++) {
    u_idx[i] = i;
    u_val[i] = (PetscScalar)i + 1.;
  }
  ierr = VecSetValues(u, 4, (const PetscInt*)u_idx, (const PetscScalar*)u_val, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Original U vector...\n");
  ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscFree(u_idx);
  ierr = PetscFree(u_val);

  ierr = PetscMalloc1(2, &v_idx);
  ierr = PetscMalloc1(2, &v_val);
  for (i=0; i<2; i++) {
    v_idx[i] = i;
    v_val[i] = (PetscScalar)i + 5.;
  }
  ierr = VecSetValues(v, 2, (const PetscInt*)v_idx, (const PetscScalar*)v_val, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Original V vector...\n");
  ierr = VecView(v, PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscFree(v_idx);
  ierr = PetscFree(v_val);

  ierr = PetscMalloc1(6, &w_idx);
  ierr = PetscMalloc1(6, &w_val);
  for (i=0; i<6; i++) {
    w_idx[i] = i;
    w_val[i] = (PetscScalar)i + 1.;
  }
  ierr = VecSetValues(w, 6, (const PetscInt*)w_idx, (const PetscScalar*)w_val, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(w);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(w);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Expected W = [U, V] vector...\n");
  ierr = VecView(w, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFree(w_idx);
  ierr = PetscFree(w_val);
  
  /* ---------- base VecConcatenate() test ----------- */
  ierr = VecConcatenate(u, v, &w_test, &u_is, &v_is);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for W = [U, W]...\n");
  ierr = VecView(w_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  w_equal = PETSC_FALSE;
  ierr = VecEqual(w_test, w, &w_equal);CHKERRQ(ierr);
  if (!w_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");
  }

  /* ---------- using index sets on expected W instead of concatenated W ----------- */
  ierr = VecGetSubVector(w, u_is, &u_test);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing index set for U component...\n");
  ierr = VecView(u_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  u_equal = PETSC_FALSE;
  ierr = VecEqual(u_test, u, &u_equal);
  if (!u_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");
  }
  ierr = VecRestoreSubVector(w, u_is, &u_test);CHKERRQ(ierr);

  ierr = VecGetSubVector(w, v_is, &v_test);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing index set for V component...\n");
  ierr = VecView(v_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  v_equal = PETSC_FALSE;
  ierr = VecEqual(v_test, v, &v_equal);
  if (!v_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");
  }
  ierr = VecRestoreSubVector(w, v_is, &v_test);CHKERRQ(ierr);
  
  /* ---------- using VecScatter to communicate data from W to U and V ----------- */
  ierr = VecDuplicate(u, &u_test);CHKERRQ(ierr);
  ierr = VecZeroEntries(u_test);CHKERRQ(ierr);
  ierr = VecScatterCreate(w_test, u_is, u, NULL, &w_to_u);CHKERRQ(ierr);
  ierr = VecScatterBegin(w_to_u, w_test, u_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(w_to_u, w_test, u_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for W -> U...\n");CHKERRQ(ierr);
  ierr = VecView(u_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  u_equal = PETSC_FALSE;
  ierr = VecEqual(u_test, u, &u_equal);CHKERRQ(ierr);
  if (!u_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");
  }

  ierr = VecDuplicate(v, &v_test);CHKERRQ(ierr);
  ierr = VecZeroEntries(v_test);CHKERRQ(ierr);
  ierr = VecScatterCreate(w_test, v_is, v, NULL, &w_to_v);CHKERRQ(ierr);
  ierr = VecScatterBegin(w_to_v, w_test, v_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(w_to_v, w_test, v_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for W -> V...\n");CHKERRQ(ierr);
  ierr = VecView(v_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  v_equal = PETSC_FALSE;
  ierr = VecEqual(v_test, v, &v_equal);CHKERRQ(ierr);
  if (!v_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");
  }

  ierr = VecZeroEntries(w_test);CHKERRQ(ierr);
  ierr = VecScatterBegin(w_to_u, u, w_test, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(w_to_u, u, w_test, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(w_to_v, v, w_test, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(w_to_v, v, w_test, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for (U, V) -> W...\n");
  ierr = VecView(w_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  w_equal = PETSC_FALSE;
  ierr = VecEqual(w_test, w, &w_equal);
  if (!w_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");
  }

  ierr = VecDestroy(&u);
  ierr = VecDestroy(&u_test);
  ierr = VecDestroy(&v);
  ierr = VecDestroy(&u_test);
  ierr = VecDestroy(&w);
  ierr = VecDestroy(&w_test);
  ierr = ISDestroy(&u_is);
  ierr = ISDestroy(&v_is);
  ierr = VecScatterDestroy(&w_to_u);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&w_to_v);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
        suffix: serial

    test:
        suffix: parallel
        nsize: 2

    test:
        suffix: cuda
        nsize: 2
        args: -vec_type cuda
        requires: cuda

TEST*/
