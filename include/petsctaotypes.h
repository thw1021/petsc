#pragma once

/* SUBMANSEC = Tao */

/*S
   Tao - Abstract PETSc object that manages nonlinear optimization solves

   Level: advanced

.seealso: [](doc_taosolve), [](ch_tao), `TaoCreate()`, `TaoDestroy()`, `TaoSetType()`, `TaoType`
S*/
typedef struct _p_Tao *Tao;

/*S
   TaoTerm - Abstract PETSc object for a parametric real-valued function that can be a term in a `Tao` objective function

   Level: intermediate

.seealso: [](doc_taosolve), [](ch_tao), `TaoTermCreate()`, `TaoTermDestroy()`, `TaoTermSetType()`, `TaoTermType`
S*/
typedef struct _p_TaoTerm *TaoTerm;
