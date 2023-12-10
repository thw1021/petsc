static char help[] = "Estimate eigenvalues with KSP.\n\n";

/*
    Test example that demonstrates how KSP can estimate eigenvalues.

    Contributed by: Pablo Brubeck <brubeck@protonmail.com>
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b; /* approx solution, RHS */
  Mat         A;    /* linear system matrix */
  KSP         ksp;  /* linear solver context */
  PC          pc;   /* preconditioner context */
  PetscInt    n = 10;
  PetscMPIInt size;
  PetscScalar data[100] = {0.01900677733082856,    0.0006930942519647015,   -0.00044241746554721565, 0.0002103712786878531,  0.0006930942519647825,   0.0006140114285409819,  -0.00046851475217568475, -0.00044241746554723034, -0.00046851475217569636,
                           0.00021037127868786035, 0.000693094251964701,    0.059270340414212176,    0.0011762392616009715,  -0.000442417465547234,   -0.0013280848558196233, -0.003216261971328536,   0.0007214671628728421,   0.000721467162872908,
                           0.0016802345162577742,  -0.00046851475217570595, -0.00044241746554721554, 0.0011762392616009706,  0.059270340414212135,    0.0006930942519647157,  0.0007214671628728805,   -0.003216261971328582,   -0.001328084855819515,
                           0.001680234516257721,   0.0007214671628728718,   -0.00046851475217569397, 0.00021037127868785333, -0.00044241746554723397, 0.000693094251964717,   0.01900677733082846,     -0.0004685147521757005,  0.000614011428541031,
                           0.0006930942519647617,  -0.000468514752175688,   -0.0004424174655472671,  0.0002103712786878798,  0.0006930942519647826,   -0.0013280848558196211, 0.0007214671628728808,   -0.0004685147521757,     0.05927034041421214,
                           -0.003216261971328475,  0.0016802345162577302,   0.001176239261600886,    0.0007214671628729087,  -0.00044241746554721283, 0.0006140114285409821,  -0.003216261971328536,   -0.0032162619713285816,  0.0006140114285410324,
                           -0.0032162619713284766, 0.10579258903720012,     -0.0032162619713285135,  -0.003216261971328439,  -0.003216261971328572,   0.000614011428540997,   -0.00046851475217568475, 0.0007214671628728421,   -0.0013280848558195133,
                           0.000693094251964762,   0.001680234516257731,    -0.0032162619713285118,  0.059270340414212246,   0.0007214671628728379,   0.0011762392616010023,  -0.00044241746554726395, -0.0004424174655472301,  0.0007214671628729061,
                           0.0016802345162577207,  -0.0004685147521756878,  0.0011762392616008856,   -0.0032162619713284354, 0.0007214671628728359,   0.05927034041421215,    -0.0013280848558195365,  0.0006930942519647073,   -0.0004685147521756969,
                           0.001680234516257775,   0.0007214671628728721,   -0.0004424174655472671,  0.0007214671628729094,  -0.003216261971328571,   0.0011762392616010058,  -0.0013280848558195398,  0.05927034041421213,     0.0006930942519648116,
                           0.0002103712786878602,  -0.00046851475217570617, -0.0004685147521756934,  0.00021037127868787982, -0.0004424174655472142,  0.000614011428540997,   -0.00044241746554726417, 0.0006930942519647065,   0.0006930942519648106,
                           0.019006777330828416};
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, data, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Set random right-hand-side vector.
  */
  PetscCall(VecSetRandom(b, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-15, 1.e-15, PETSC_DEFAULT, PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Solve linear system
  */
  PetscCall(KSPSolve(ksp, b, x));

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -ksp_type cg -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor_short -ksp_converged_reason

   test:
      suffix: 2
      args: -ksp_type fcg -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor_short -ksp_converged_reason

   test:
      suffix: 3
      args: -ksp_type minres -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor_short -ksp_converged_reason

   test:
      suffix: 4
      args: -ksp_type gmres -ksp_pc_side left -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor_short -ksp_converged_reason

   test:
      suffix: 5
      args: -ksp_type gmres -ksp_pc_side right -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor_short -ksp_converged_reason

   test:
      suffix: 6
      args: -ksp_type fgmres -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor_short -ksp_converged_reason

TEST*/
