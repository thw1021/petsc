static char help[] = "Demonstrates use of VecLocalFormSetIS().\n\n";

/*
   Description: Ghost padding is one way to handle local calculations that
      involve values from other processors. VecLocalFormSetIS() provides
      a way to create vectors with extra room at the end of the vector
      array to contain the needed ghost values from other processors,
      vector computations are otherwise unaffected.
*/

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscMPIInt        rank, size;
  PetscInt           nlocal = 6, nghost = 2, ifrom[2], i, rstart, rend;
  PetscScalar       *warray;
  const PetscScalar *rarray;
  Vec                lx, gx;
  IS                 ghosts;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run example with two processors");

  /*
     Construct a two dimensional graph connecting nlocal degrees of
     freedom per processor. From this we will generate the global
     indices of needed ghost values

     For simplicity we generate the entire graph on each processor:
     in real application the graph would stored in parallel, but this
     example is only to demonstrate the management of ghost padding
     with VecLocalFormSetIS().

     In this example we consider the vector as representing
     degrees of freedom in a one dimensional grid with periodic
     boundary conditions.

        ----Processor  1---------  ----Processor 2 --------
         0    1   2   3   4    5    6    7   8   9   10   11
                               |----|
         |-------------------------------------------------|

  */

  /*
     Create the vector with two slots for ghost points. Note that both
     the local vector (lx) and the global vector (gx) share the same
     array for storing vector values.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &gx));
  PetscCall(VecSetType(gx, VECMPI));
  PetscCall(VecSetSizes(gx, nlocal, PETSC_DECIDE));

  if (rank == 0) {
    ifrom[0] = 11;
    ifrom[1] = 6;
  } else {
    ifrom[0] = 0;
    ifrom[1] = 5;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nghost, ifrom, PETSC_COPY_VALUES, &ghosts));
  PetscCall(VecLocalFormSetIS(gx, ghosts));
  PetscCall(ISDestroy(&ghosts));

  /*
     Set the values from 0 to 12 into the "global" vector
  */
  PetscCall(VecGetOwnershipRange(gx, &rstart, &rend));
  for (i = rstart; i < rend; i++) { PetscCall(VecSetValue(gx, i, (PetscScalar)i, INSERT_VALUES)); }
  PetscCall(VecAssemblyBegin(gx));
  PetscCall(VecAssemblyEnd(gx));

  /*
     Access the local representation and print it out, including the ghost padding region.
  */
  PetscCall(VecLocalFormGetRead(gx, &lx));

  PetscCall(VecGetArrayRead(lx, &rarray));
  for (i = 0; i < nlocal + nghost; i++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " %g\n", i, (double)PetscRealPart(rarray[i])));
  PetscCall(VecRestoreArrayRead(lx, &rarray));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(VecLocalFormRestoreRead(gx, &lx));

  /* Set ghost values and then accumulates onto the owning processors */
  PetscCall(VecLocalFormGetWrite(gx, &lx));
  PetscCall(VecGetArray(lx, &warray));
  for (i = 0; i < nghost; i++) warray[nlocal + i] = rank ? (PetscScalar)4 : (PetscScalar)8;
  PetscCall(VecRestoreArray(lx, &warray));
  PetscCall(VecLocalFormRestoreWrite(gx, ADD_VALUES, &lx));
  PetscCall(VecView(gx, PETSC_VIEWER_STDOUT_WORLD));

  /* Check VecSetValuesLocal() */
  PetscCall(VecSetValueLocal(gx, 6, 20, ADD_VALUES));
  PetscCall(VecAssemblyBegin(gx));
  PetscCall(VecAssemblyEnd(gx));
  PetscCall(VecView(gx, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&gx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
