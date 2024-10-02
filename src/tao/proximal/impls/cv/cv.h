#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  Mat h_lmap;                                                /* m * n               */
  Vec workvec, workvec2, grad_old, x_old, ATy;               /* size n              */
  Vec dualvec_work, dualvec_test, dualvec_work2, Ax, Ax_old; /* size m. dualvec = y */
  Vec g_param, h_param;                                      //TODO no coverage test for g_param

  PetscReal     step_old;
  PetscReal     gnorm_norm;
  PetscReal     h_lmap_norm;
  PetscReal     f_scale;
  PetscReal     g_scale;
  PetscReal     h_scale;
  PetscReal     lip;
  PetscReal     rho;
  PetscReal     sigma;
  PetscReal     eta;      // lmap_norm estimate
  PetscReal     Theta;    // usually 1.1 <= Theta <= 1.5
  PetscReal     pd_ratio; //t variable
  PetscReal     R;        //scale factor for estimating linear map norm. Must be <= 1.
  PetscReal     r;        //backtracking parameter > 1
  PetscBool     lip_set, lmap_norm_set, approx_lmap_norm;
  TaoTerm       h_cj_term;
  TaoMappedTerm reg_term;
  TaoMappedTerm f_term;
  TaoMappedTerm g_term;
  TaoMappedTerm h_term;
  TaoMappedTerm h_cj_mapped_term; // TaoTermCreateConvexConjugate(h_term.term, &h_dual_term.term);
} TAO_CV;
