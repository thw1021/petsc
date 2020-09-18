#if !defined(PCFIELDSPLITIMPL_H)
#define PCFIELDSPLITIMPL_H

#include <petscdm.h>

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  KSP               ksp;
  Vec               x,y,z;
  char              *splitname;
  PetscInt          nfields;
  PetscInt          *fields,*fields_col;
  VecScatter        sctx;
  IS                is,is_col;
  PC_FieldSplitLink next,previous;
  PetscLogEvent     event;
};

typedef struct {
  PCCompositeType type;
  PetscBool       defaultsplit;                    /* Flag for a system with a set of 'k' scalar fields with the same layout (and bs = k) */
  PetscBool       splitdefined;                    /* Flag is set after the splits have been defined, to prevent more splits from being added */
  PetscInt        bs;                              /* Block size for IS and Mat structures */
  PetscInt        nsplits;                         /* Number of field divisions defined */
  Vec             *x,*y,w1,w2;
  Mat             *mat;                            /* The diagonal block for each split */
  Mat             *pmat;                           /* The preconditioning diagonal block for each split */
  Mat             *Afield;                         /* The rows of the matrix associated with each split */
  PetscBool       issetup;

  /* Only used when Schur complement preconditioning is used */
  Mat                       B;                     /* The (0,1) block */
  Mat                       C;                     /* The (1,0) block */
  Mat                       schur;                 /* The Schur complement S = A11 - A10 A00^{-1} A01, the KSP here, kspinner, is H_1 in [El08] */
  Mat                       schurp;                /* Assembled approximation to S built by MatSchurComplement to be used as a preconditioning matrix when solving with S */
  Mat                       schur_user;            /* User-provided preconditioning matrix for the Schur complement */
  PCFieldSplitSchurPreType  schurpre;              /* Determines which preconditioning matrix is used for the Schur complement */
  PCFieldSplitSchurFactType schurfactorization;
  KSP                       kspschur;              /* The solver for S */
  KSP                       kspupper;              /* The solver for A in the upper diagonal part of the factorization (H_2 in [El08]) */
  PetscScalar               schurscale;            /* Scaling factor for the Schur complement solution with DIAG factorization */

  /* Only used when Golub-Kahan bidiagonalization preconditioning is used */
  Mat                       H;                     /* The modified matrix H = A00 + nu*A01*A01'              */
  PetscReal                 gkbtol;                /* Stopping tolerance for lower bound estimate            */
  PetscInt                  gkbdelay;              /* The delay window for the stopping criterion            */
  PetscReal                 gkbnu;                 /* Parameter for augmented Lagrangian H = A + nu*A01*A01' */
  PetscInt                  gkbmaxit;              /* Maximum number of iterations for outer loop            */
  PetscBool                 gkbmonitor;            /* Monitor for gkb iterations and the lower bound error   */
  PetscViewer               gkbviewer;             /* Viewer context for gkbmonitor                          */
  Vec                       u,v,d,Hu;              /* Work vectors for the GKB algorithm                     */
  PetscScalar               *vecz;                 /* Contains intermediate values, eg for lower bound       */

  PC_FieldSplitLink         head;
  PetscBool                 isrestrict;             /* indicates PCFieldSplitRestrictIS() has been last called on this object, hack */
  PetscBool                 suboptionsset;          /* Indicates that the KSPSetFromOptions() has been called on the sub-KSPs */
  PetscBool                 dm_splits;              /* Whether to use DMCreateFieldDecomposition() whenever possible */
  PetscBool                 diag_use_amat;          /* Whether to extract diagonal matrix blocks from Amat, rather than Pmat (weaker than -pc_use_amat) */
  PetscBool                 offdiag_use_amat;       /* Whether to extract off-diagonal matrix blocks from Amat, rather than Pmat (weaker than -pc_use_amat) */
  PetscBool                 detect;                 /* Whether to form 2-way split by finding zero diagonal entries */
} PC_FieldSplit;


#endif /* PCFIELDSPLITIMPL_H */
