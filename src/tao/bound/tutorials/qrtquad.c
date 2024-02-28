const char help[] = "Implementation of QRTQUAD bound constrained optimization problem";

/// QRTQUAD - The objective function is
//
// f(x) = \sum_{i = 1}^n x_i * (-10 * i) + \sum_{i = 1}^m (i / m) * (x_i * x_{i+1})^4 + \sum_{i=m+1}^{n-1} 4 x_i^2 + 2 * x_n^2 + x_i * x_n
//        \____________________________/   \________________________________________/   \________________________________________________/
//                       |                                     |                                                |
//                 linear terms     "quartic" terms (really degree 8, quartic in x_i * x_{i+1})          quadratic terms
//
// the variables (x_1, ..., x_m) are bounded in [0, 10]

#include <petsctao.h>
#include <petscsf.h>
#include <petscdevice.h>
#include <petscdevice_cupm.h>

typedef struct _AppCtx *AppCtx;

struct _AppCtx {
  MPI_Comm    comm;
  PetscInt    n; // Total number of degrees of freedom
  PetscInt    m; // Number of quartic & bounded degrees of freedeom
  PetscInt    k; // Number of copies
  PetscInt    r_start;
  PetscInt    r_end;
  PetscInt    q_start;
  PetscInt    q_end;
  Vec         x;      // solution
  Vec         x_next; // k values from the next process
  VecScatter  x_next_scatter;
  Vec         x_end; // k values from the end of the vector
  PetscSF     x_end_bcast;
  Vec         xl; // lower bound
  Vec         xu; // upper bound
  Vec         g;  // gradient
  Vec         g_end;
  Vec         g_end_local;
  Vec         g_end_ones;
  Mat         g_end_mat; // gradient
  Vec         g_lin;     // from the linear term
  Vec         g_y_qrt;
  VecScatter  g_y_qrt_scatter;
  Vec         vcopy; // local copies of the global v variables
  Vec         gvals; // gradient entries (excluding last k variables)
  Mat         H;     // Hessian preconditioner
  Vec         Hvals; // Hessian entries
  PetscLayout layout;
  PetscBool   set_from_options_called;
  PetscBool   owns_end;
};

static PetscErrorCode AppCtxCreate(MPI_Comm comm, AppCtx *ctx)
{
  PetscMPIInt size;
  AppCtx      c;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscNew(ctx));
  c       = *ctx;
  c->comm = comm;
  c->n    = 5000;
  c->m    = 1100;
  c->k    = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxSetFromOptions(AppCtx ctx)
{
  PetscFunctionBegin;
  PetscOptionsBegin(ctx->comm, NULL, help, NULL);
  PetscCall(PetscOptionsBoundedInt("-n", "number of optimization variables", NULL, ctx->n, &ctx->n, NULL, 0));
  ctx->m = PetscMin(ctx->m, ctx->n);
  PetscCall(PetscOptionsRangeInt("-m", "number of quartic and bounded variables", NULL, ctx->m, &ctx->m, NULL, 0, ctx->n));
  PetscCall(PetscOptionsBoundedInt("-k", "number of copies", NULL, ctx->k, &ctx->k, NULL, 1));
  PetscOptionsEnd();
  ctx->set_from_options_called = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxSetUp(AppCtx ctx)
{
  PetscInt     m, n, k, r_start, r_end;
  PetscScalar *ga, *la, *ua;
  PetscInt     q_start, q_end;
  PetscInt    *h_i, *h_j;
  VecType      vec_type;

  PetscFunctionBegin;
  n = ctx->n;
  m = ctx->m;
  k = ctx->k;

  // Layout of solution degrees of freedom
  PetscCall(PetscLayoutCreate(ctx->comm, &ctx->layout));
  PetscCall(PetscLayoutSetSize(ctx->layout, n * k));
  PetscCall(PetscLayoutSetLocalSize(ctx->layout, PETSC_DETERMINE));
  PetscCall(PetscLayoutSetBlockSize(ctx->layout, k));
  PetscCall(PetscLayoutSetUp(ctx->layout));
  PetscCall(PetscLayoutGetRange(ctx->layout, &r_start, &r_end));
  ctx->r_start = r_start;
  ctx->r_end   = r_end;
  PetscCheck(r_start % k == 0, ctx->comm, PETSC_ERR_PLIB, "k boundary not respected");
  PetscCheck(r_end % k == 0, ctx->comm, PETSC_ERR_PLIB, "k boundary not respected");

  // Hessian: allocated even if it isn't used
  PetscCall(MatCreate(ctx->comm, &ctx->H));
  PetscCall(MatSetLayouts(ctx->H, ctx->layout, ctx->layout));
  PetscCall(MatSetBlockSize(ctx->H, ctx->k));
  PetscCall(MatSetType(ctx->H, MATAIJ));
  if (ctx->set_from_options_called) PetscCall(MatSetFromOptions(ctx->H));

  // COO Preallocation for H
  {
    ctx->q_start = q_start = r_start / k;
    ctx->q_end = q_end = PetscMax(q_start, PetscMin(n - 1, r_end / k));
    PetscCall(PetscMalloc2(4 * k * (q_end - q_start), &h_i, 4 * k * (q_end - q_start), &h_j));

    PetscInt *h_i_xx = &h_i[0 * k * (q_end - q_start)];
    PetscInt *h_i_xy = &h_i[1 * k * (q_end - q_start)];
    PetscInt *h_i_yx = &h_i[2 * k * (q_end - q_start)];
    PetscInt *h_i_yy = &h_i[3 * k * (q_end - q_start)];
    PetscInt *h_j_xx = &h_j[0 * k * (q_end - q_start)];
    PetscInt *h_j_xy = &h_j[1 * k * (q_end - q_start)];
    PetscInt *h_j_yx = &h_j[2 * k * (q_end - q_start)];
    PetscInt *h_j_yy = &h_j[3 * k * (q_end - q_start)];
    // Tridiagonal structure for the quartic terms
    for (PetscInt r = r_start; r < k * PetscMin(m, q_end); r++) {
      h_i_xx[r - r_start] = r;
      h_i_xy[r - r_start] = r;
      h_i_yx[r - r_start] = r + k;
      h_i_yy[r - r_start] = r + k;
      h_j_xx[r - r_start] = r;
      h_j_xy[r - r_start] = r + k;
      h_j_yx[r - r_start] = r;
      h_j_yy[r - r_start] = r + k;
    }
    // Arrowhead structrure for the quadratic terms
    for (PetscInt q = PetscMax(q_start, PetscMin(m, q_end)); q < q_end; q++) {
      for (PetscInt i = 0; i < k; i++) {
        PetscInt r = i + k * q;
        PetscInt s = i + k * (n - 1);

        h_i_xx[r - r_start] = r;
        h_i_xy[r - r_start] = r;
        h_i_yx[r - r_start] = s;
        h_i_yy[r - r_start] = s;
        h_j_xx[r - r_start] = r;
        h_j_xy[r - r_start] = s;
        h_j_yx[r - r_start] = r;
        h_j_yy[r - r_start] = s;
      }
    }
    PetscCall(MatSetPreallocationCOO(ctx->H, 4 * k * (q_end - q_start), h_i, h_j));
    PetscCall(PetscFree2(h_i, h_j));
  }
  PetscCall(MatSetUp(ctx->H));
  PetscCall(MatZeroEntries(ctx->H));
  PetscCall(MatAssemblyBegin(ctx->H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(ctx->H, NULL, "-hessian_structure_view"));

  PetscCall(MatCreateVecs(ctx->H, &ctx->x, &ctx->g));
  PetscCall(VecZeroEntries(ctx->x));
  PetscCall(VecGetType(ctx->x, &vec_type));

  PetscCall(VecCreate(ctx->comm, &ctx->Hvals));
  PetscCall(VecSetSizes(ctx->Hvals, 4 * k * (q_end - q_start), PETSC_DETERMINE));
  PetscCall(VecSetType(ctx->Hvals, vec_type));
  PetscCall(VecSetUp(ctx->Hvals));

  // the quartic terms need ghost data
  {
    PetscInt n_next = 0;
    IS       next_nodes_is;

    // This process had quartic terms unless
    if (r_start < r_end && r_end <= k * m) n_next = k;
    PetscCall(ISCreateStride(ctx->comm, n_next, r_end, 1, &next_nodes_is));

    PetscCall(VecCreate(ctx->comm, &ctx->x_next));
    PetscCall(VecSetSizes(ctx->x_next, n_next, PETSC_DETERMINE));
    PetscCall(VecSetType(ctx->x_next, vec_type));
    PetscCall(VecSetUp(ctx->x_next));
    PetscCall(VecScatterCreate(ctx->x, next_nodes_is, ctx->x_next, NULL, &ctx->x_next_scatter));
    PetscCall(ISDestroy(&next_nodes_is));
    PetscCall(VecScatterSetUp(ctx->x_next_scatter));
    PetscCall(VecScatterViewFromOptions(ctx->x_next_scatter, NULL, "-x_next_scatter_view"));
  }

  // the last entries need to be broadcast for the quadratic terms
  {
    PetscLayout gather_layout;
    PetscBool   owns_end = PETSC_FALSE;

    if (r_start < r_end && r_end == k * n) owns_end = PETSC_TRUE;
    ctx->owns_end = owns_end;

    PetscCall(VecCreate(ctx->comm, &ctx->x_end));
    PetscCall(VecSetSizes(ctx->x_end, k, PETSC_DETERMINE));
    PetscCall(VecSetType(ctx->x_end, vec_type));
    PetscCall(VecSetUp(ctx->x_end));

    PetscCall(PetscSFCreate(ctx->comm, &ctx->x_end_bcast));
    PetscCall(PetscLayoutCreate(ctx->comm, &gather_layout));
    PetscCall(PetscLayoutSetSize(gather_layout, k));
    PetscCall(PetscLayoutSetLocalSize(gather_layout, owns_end ? k : 0));
    PetscCall(PetscLayoutSetUp(gather_layout));
    PetscCall(PetscSFSetGraphWithPattern(ctx->x_end_bcast, gather_layout, PETSCSF_PATTERN_ALLGATHER));
    PetscCall(PetscLayoutDestroy(&gather_layout));
    PetscCall(PetscSFSetUp(ctx->x_end_bcast));
    PetscCall(VecScatterViewFromOptions(ctx->x_end_bcast, NULL, "-x_end_bcast_view"));
  }

  // create bounds for the first m variables
  PetscCall(VecDuplicate(ctx->x, &ctx->xl));
  PetscCall(VecDuplicate(ctx->x, &ctx->xu));
  PetscCall(VecSet(ctx->xl, PETSC_NINFINITY));
  PetscCall(VecSet(ctx->xu, PETSC_INFINITY));
  PetscCall(VecGetArray(ctx->xl, &la));
  PetscCall(VecGetArray(ctx->xu, &ua));
  // variables in [0, m) are bounded in [0, 10]
  for (PetscInt i = r_start; i < PetscMin(r_end, m); i++) {
    la[i - r_start] = 0.0;
    ua[i - r_start] = 10.0;
  }
  PetscCall(VecRestoreArray(ctx->xu, &ua));
  PetscCall(VecRestoreArray(ctx->xl, &la));
  PetscCall(VecViewFromOptions(ctx->xl, NULL, "-xl_view"));
  PetscCall(VecViewFromOptions(ctx->xu, NULL, "-xu_view"));

  // the vector for the linear portion of the gradient
  PetscCall(VecDuplicate(ctx->g, &ctx->g_lin));
  PetscCall(VecGetArray(ctx->g_lin, &ga));
  for (PetscInt i = r_start; i < r_end; i++) { ga[i - r_start] = -10.0 * (((PetscInt)(i / k)) + 1); }
  PetscCall(VecRestoreArray(ctx->g_lin, &ga));
  PetscCall(VecViewFromOptions(ctx->g_lin, NULL, "-linear_gradient_view"));

  PetscCall(VecDuplicate(ctx->x_end, &ctx->g_end));
  {
    PetscInt q_quad_local = PetscMax(0, q_end - PetscMax(m, q_start));

    PetscCall(MatCreateDenseFromVecType(PETSC_COMM_SELF, vec_type, k, q_quad_local, k, q_quad_local, k, NULL, &ctx->g_end_mat));
    PetscCall(VecCreateLocalVector(ctx->g_end, &ctx->g_end_local));
    PetscCall(MatCreateVecs(ctx->g_end_mat, &ctx->g_end_ones, NULL));
    PetscCall(VecSet(ctx->g_end_ones, 1.0));
  }

  PetscCall(VecCreate(ctx->comm, &ctx->g_y_qrt));
  PetscCall(VecSetSizes(ctx->g_y_qrt, PetscMax(0, k * (m - q_start)), k * m));
  PetscCall(VecSetType(ctx->g_y_qrt, vec_type));
  PetscCall(VecSetUp(ctx->g_y_qrt));

  // the quartic terms need a scatter for writing the y gradient terms
  {
    PetscInt n_y_qrt = k * PetscMax(0, PetscMin(m, q_end) - q_start);
    IS       y_qrt_is;

    PetscCall(ISCreateStride(ctx->comm, n_y_qrt, k * (q_start + 1), 1, &y_qrt_is));

    PetscCall(VecScatterCreate(ctx->g, y_qrt_is, ctx->g_y_qrt, NULL, &ctx->g_y_qrt_scatter));
    PetscCall(ISDestroy(&y_qrt_is));
    PetscCall(VecScatterSetUp(ctx->g_y_qrt_scatter));
    PetscCall(VecScatterViewFromOptions(ctx->g_y_qrt_scatter, NULL, "-g_y_qrt_scatter_view"));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QrtObjective(PetscScalar x, PetscScalar y, PetscReal p)
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a4 = a2 * a2;
  return p * a4;
}

static const PetscLogDouble QrtObjectiveFlops = 4.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QrtGradient(PetscScalar x, PetscScalar y, PetscReal p, PetscScalar *PETSC_RESTRICT g_x, PetscScalar *PETSC_RESTRICT g_y)
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a3 = a * a2;
  *g_x += y * (p * 4.0 * a3);
  *g_y = x * (p * 4.0 * a3);
}

static const PetscLogDouble QrtGradientFlops = 8.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QrtObjectiveGradient(PetscScalar x, PetscScalar y, PetscReal p, PetscScalar *PETSC_RESTRICT g_x, PetscScalar *PETSC_RESTRICT g_y)
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a3 = a * a2;
  PetscScalar a4 = a * a3;
  *g_x += y * (p * 4.0 * a3);
  *g_y = x * (p * 4.0 * a3);
  return p * a4;
}

static const PetscLogDouble QrtObjectiveGradientFlops = 10.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QrtHessian(PetscScalar x, PetscScalar y, PetscReal p, PetscScalar *PETSC_RESTRICT h_xx, PetscScalar *PETSC_RESTRICT h_xy, PetscScalar *PETSC_RESTRICT h_yx, PetscScalar *PETSC_RESTRICT h_yy)
{
  PetscScalar a  = x * y;
  PetscScalar a2 = a * a;
  PetscScalar a3 = a * a2;
  *h_xx          = (y * y) * (p * 12.0 * a2);
  *h_xy = *h_yx = (p * 16.0 * a3);
  *h_yy         = (x * x) * (p * 12.0 * a2);
}

static const PetscLogDouble QrtHessianFlops = 9.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QuadObjective(PetscScalar x, PetscScalar y)
{
  return (4.0 * x * x + 2.0 * y * y + x * y);
}

static const PetscLogDouble QuadObjectiveFlops = 8.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QuadGradient(PetscScalar x, PetscScalar y, PetscScalar *PETSC_RESTRICT g_x, PetscScalar *PETSC_RESTRICT g_y)
{
  *g_x += 8.0 * x + y;
  *g_y = 4.0 * y + x;
}

static const PetscLogDouble QuadGradientFlops = 5.0;

static PETSC_HOSTDEVICE_INLINE_DECL PetscReal QuadObjectiveGradient(PetscScalar x, PetscScalar y, PetscScalar *PETSC_RESTRICT g_x, PetscScalar *PETSC_RESTRICT g_y)
{
  *g_x += 8.0 * x + y;
  *g_y = 4.0 * y + x;
  return (4.0 * x * x + 2.0 * y * y + x * y);
}

static const PetscLogDouble QuadObjectiveGradientFlops = 13.0;

static PETSC_HOSTDEVICE_INLINE_DECL void QuadHessian(PetscScalar x, PetscScalar y, PetscScalar *PETSC_RESTRICT h_xx, PetscScalar *PETSC_RESTRICT h_xy, PetscScalar *PETSC_RESTRICT h_yx, PetscScalar *PETSC_RESTRICT h_yy)
{
  *h_xx = 8.0;
  *h_xy = *h_yx = 1.0;
  *h_yy         = 4.0;
}

static const PetscLogDouble QuadHessianFlops = 9.0;

static PetscErrorCode QrtQuadObjective_Host(AppCtx ctx, const PetscScalar *PETSC_RESTRICT X, const PetscScalar *PETSC_RESTRICT X_next, const PetscScalar *PETSC_RESTRICT X_end, PetscReal *f)
{
  PetscReal _f = 0.0;
  PetscInt  m, k, q_start, q_end, q_mid, q_mid_local;

  PetscFunctionBegin;
  m           = ctx->m;
  k           = ctx->k;
  q_start     = ctx->q_start;
  q_end       = ctx->q_end;
  q_mid       = PetscMax(q_start, PetscMin(m, q_end));
  q_mid_local = q_mid;

  // quartic terms
  if (q_mid == q_end) q_mid_local = PetscMax(q_start, q_mid - 1);
  for (PetscInt q = q_start; q < q_mid_local; q++) {
    PetscReal p = ((PetscReal)q + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q - q_start);

      _f += QrtObjective(X[j], X[j + k], p);
    }
  }
  if (q_mid_local + 1 == q_mid) {
    PetscReal p = ((PetscReal)q_mid_local + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q_mid_local - q_start);

      _f += QrtObjective(X[j], X_next[i], p);
    }
  }
  PetscCall(PetscLogFlops((QrtObjectiveFlops + 1) * (q_mid - q_start) * k));

  // quadratic terms
  for (PetscInt q = q_mid; q < q_end; q++) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q - q_start);

      _f += QuadObjective(X[j], X_end[i]);
    }
  }
  PetscCall(PetscLogFlops((QuadObjectiveFlops + 1) * (q_end - q_mid) * k));
  *f += _f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadGradient_Host(AppCtx ctx, const PetscScalar *PETSC_RESTRICT X, const PetscScalar *PETSC_RESTRICT X_next, const PetscScalar *PETSC_RESTRICT X_end, PetscScalar *PETSC_RESTRICT g_x, PetscScalar *PETSC_RESTRICT g_y_qrt, PetscScalar *PETSC_RESTRICT g_y_quad)
{
  PetscInt m, k, q_start, q_end, q_mid, q_mid_local;

  PetscFunctionBegin;
  m           = ctx->m;
  k           = ctx->k;
  q_start     = ctx->q_start;
  q_end       = ctx->q_end;
  q_mid       = PetscMax(q_start, PetscMin(m, q_end));
  q_mid_local = q_mid;

  // quartic terms
  if (q_mid == q_end) q_mid_local = PetscMax(q_start, q_mid - 1);
  for (PetscInt q = q_start; q < q_mid_local; q++) {
    PetscReal p = ((PetscReal)q + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q - q_start);

      QrtGradient(X[j], X[j + k], p, &g_x[j], &g_y_qrt[j]);
    }
  }
  if (q_mid_local + 1 == q_mid) {
    PetscReal p = ((PetscReal)q_mid_local + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q_mid_local - q_start);

      QrtGradient(X[j], X_next[i], p, &g_x[j], &g_y_qrt[j]);
    }
  }
  PetscCall(PetscLogFlops(QrtGradientFlops * (q_mid - q_start) * k));

  // quadratic terms
  for (PetscInt q = q_mid; q < q_end; q++) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j      = i + k * (q - q_start);
      PetscInt j_quad = i + k * (q - q_mid);

      QuadGradient(X[j], X_end[i], &g_x[j], &g_y_quad[j_quad]);
    }
  }
  PetscCall(PetscLogFlops(QuadGradientFlops * (q_end - q_mid) * k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadObjectiveGradient_Host(AppCtx ctx, const PetscScalar *PETSC_RESTRICT X, const PetscScalar *PETSC_RESTRICT X_next, const PetscScalar *PETSC_RESTRICT X_end, PetscScalar *PETSC_RESTRICT g_x, PetscScalar *PETSC_RESTRICT g_y_qrt, PetscScalar *PETSC_RESTRICT g_y_quad, PetscReal *f)
{
  PetscReal _f = 0.0;
  PetscInt  m, k, q_start, q_end, q_mid, q_mid_local;

  PetscFunctionBegin;
  m           = ctx->m;
  k           = ctx->k;
  q_start     = ctx->q_start;
  q_end       = ctx->q_end;
  q_mid       = PetscMax(q_start, PetscMin(m, q_end));
  q_mid_local = q_mid;

  // quartic terms
  if (q_mid == q_end) q_mid_local = PetscMax(q_start, q_mid - 1);
  for (PetscInt q = q_start; q < q_mid_local; q++) {
    PetscReal p = ((PetscReal)q + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q - q_start);

      _f += QrtObjectiveGradient(X[j], X[j + k], p, &g_x[j], &g_y_qrt[j]);
    }
  }
  if (q_mid_local + 1 == q_mid) {
    PetscReal p = ((PetscReal)q_mid_local + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q_mid_local - q_start);

      _f += QrtObjectiveGradient(X[j], X_next[i], p, &g_x[j], &g_y_qrt[j]);
    }
  }
  PetscCall(PetscLogFlops((QrtObjectiveGradientFlops + 1) * (q_mid - q_start) * k));

  // quadratic terms
  for (PetscInt q = q_mid; q < q_end; q++) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j      = i + k * (q - q_start);
      PetscInt j_quad = i + k * (q - q_mid);

      _f += QuadObjectiveGradient(X[j], X_end[i], &g_x[j], &g_y_quad[j_quad]);
    }
  }
  PetscCall(PetscLogFlops((QuadObjectiveGradientFlops + 1) * (q_end - q_mid) * k));
  *f += _f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadHessian_Host(AppCtx ctx, const PetscScalar *PETSC_RESTRICT X, const PetscScalar *PETSC_RESTRICT X_next, const PetscScalar *PETSC_RESTRICT X_end, PetscScalar *PETSC_RESTRICT h_xx, PetscScalar *PETSC_RESTRICT h_xy, PetscScalar *PETSC_RESTRICT h_yx, PetscScalar *PETSC_RESTRICT h_yy)
{
  PetscInt m, k, q_start, q_end, q_mid, q_mid_local;

  PetscFunctionBegin;
  m           = ctx->m;
  k           = ctx->k;
  q_start     = ctx->q_start;
  q_end       = ctx->q_end;
  q_mid       = PetscMax(q_start, PetscMin(m, q_end));
  q_mid_local = q_mid;

  // quartic terms
  if (q_mid == q_end) q_mid_local = PetscMax(q_start, q_mid - 1);
  for (PetscInt q = q_start; q < q_mid_local; q++) {
    PetscReal p = ((PetscReal)q + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q - q_start);

      QrtHessian(X[j], X[j + k], p, &h_xx[j], &h_xy[j], &h_yx[j], &h_yy[j]);
    }
  }
  if (q_mid_local + 1 == q_mid) {
    PetscReal p = ((PetscReal)q_mid_local + 1.0) * (1.0 / (PetscReal)m);

    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q_mid_local - q_start);

      QrtHessian(X[j], X_next[i], p, &h_xx[j], &h_xy[j], &h_yx[j], &h_yy[j]);
    }
  }
  PetscCall(PetscLogFlops((QrtHessianFlops + 1) * (q_mid - q_start) * k));

  // quadratic terms
  for (PetscInt q = q_mid; q < q_end; q++) {
    PetscPragmaSIMD
    for (PetscInt i = 0; i < k; i++) {
      PetscInt j = i + k * (q - q_start);

      QuadHessian(X[j], X_end[i], &h_xx[j], &h_xy[j], &h_yx[j], &h_yy[j]);
    }
  }
  PetscCall(PetscLogFlops((QuadHessianFlops + 1) * (q_end - q_mid) * k));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxGetSolutionArraysRead(AppCtx ctx, Vec x, const PetscScalar **x_a, const PetscScalar **x_next_a, const PetscScalar **x_end_a)
{
  PetscInt           k, r_start, r_end;
  const PetscScalar *xa, *xnexta, *xa_root = NULL;
  PetscScalar       *xenda;

  PetscFunctionBegin;
  r_start = ctx->r_start;
  r_end   = ctx->r_end;
  k       = ctx->k;
  PetscCall(VecScatterBegin(ctx->x_next_scatter, x, ctx->x_next, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->x_next_scatter, x, ctx->x_next, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArrayRead(ctx->x_next, &xnexta));
  PetscCall(VecGetArray(ctx->x_end, &xenda));
  if (ctx->owns_end) xa_root = &xa[(r_end - r_start) - k];
  PetscCall(PetscSFBcastBegin(ctx->x_end_bcast, MPIU_SCALAR, xa_root, xenda, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(ctx->x_end_bcast, MPIU_SCALAR, xa_root, xenda, MPI_REPLACE));
  *x_a      = xa;
  *x_next_a = xnexta;
  *x_end_a  = xenda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxRestoreSolutionArraysRead(AppCtx ctx, Vec x, const PetscScalar **x_a, const PetscScalar **x_next_a, const PetscScalar **x_end_a)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreArray(ctx->x_end, (PetscScalar **)x_end_a));
  PetscCall(VecRestoreArrayRead(ctx->x_next, x_next_a));
  PetscCall(VecRestoreArrayRead(x, x_a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxGetGradientArrays(AppCtx ctx, Vec g, PetscScalar **g_x, PetscScalar **g_y_qrt, PetscScalar **g_y_quad)
{
  PetscFunctionBegin;
  PetscCall(VecGetArray(g, g_x));
  PetscCall(VecGetArrayWrite(ctx->g_y_qrt, g_y_qrt));
  PetscCall(MatDenseGetArrayWrite(ctx->g_end_mat, g_y_quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxRestoreGradientArrays(AppCtx ctx, Vec g, PetscScalar **g_x, PetscScalar **g_y_qrt, PetscScalar **g_y_quad)
{
  PetscInt           k, r_start, r_end;
  const PetscScalar *g_end_a;
  PetscScalar       *g_end_root_a = NULL;

  PetscFunctionBegin;
  r_start = ctx->r_start;
  r_end   = ctx->r_end;
  k       = ctx->k;
  PetscCall(MatDenseRestoreArrayWrite(ctx->g_end_mat, g_y_quad));
  PetscCall(VecGetLocalVector(ctx->g_end, ctx->g_end_local));
  PetscCall(MatMult(ctx->g_end_mat, ctx->g_end_ones, ctx->g_end_local));
  PetscCall(VecRestoreLocalVector(ctx->g_end, ctx->g_end_local));
  PetscCall(VecGetArrayRead(ctx->g_end, &g_end_a));
  if (ctx->owns_end) g_end_root_a = &((*g_x)[(r_end - r_start) - k]);
  PetscCall(PetscSFReduceBegin(ctx->x_end_bcast, MPIU_SCALAR, g_end_a, g_end_root_a, MPI_SUM));
  PetscCall(PetscSFReduceEnd(ctx->x_end_bcast, MPIU_SCALAR, g_end_a, g_end_root_a, MPI_SUM));
  PetscCall(VecRestoreArrayRead(ctx->g_end, &g_end_a));
  PetscCall(VecRestoreArrayWrite(ctx->g_y_qrt, g_y_qrt));
  PetscCall(VecRestoreArray(g, g_x));
  PetscCall(VecScatterBegin(ctx->g_y_qrt_scatter, ctx->g_y_qrt, g, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->g_y_qrt_scatter, ctx->g_y_qrt, g, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxGetHessianArraysWrite(AppCtx ctx, Mat H, PetscScalar **h_xx, PetscScalar **h_xy, PetscScalar **h_yx, PetscScalar **h_yy)
{
  PetscScalar *ha;
  PetscInt     k, q_start, q_end;

  PetscFunctionBegin;
  q_start = ctx->q_start;
  q_end   = ctx->q_end;
  k       = ctx->k;
  PetscCall(VecGetArrayWrite(ctx->Hvals, &ha));
  *h_xx = &ha[0 * k * (q_end - q_start)];
  *h_xy = &ha[1 * k * (q_end - q_start)];
  *h_yx = &ha[2 * k * (q_end - q_start)];
  *h_yy = &ha[3 * k * (q_end - q_start)];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxRestoreHessianArraysWrite(AppCtx ctx, Mat H, PetscScalar **h_xx, PetscScalar **h_xy, PetscScalar **h_yx, PetscScalar **h_yy)
{
  PetscFunctionBegin;
  PetscCall(MatSetValuesCOO(H, *h_xx, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(H, NULL, "-hessian_view"));
  PetscCall(VecRestoreArrayWrite(ctx->Hvals, h_xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadObjective(Tao tao, Vec x, PetscReal *f, void *data)
{
  AppCtx             ctx = (AppCtx)data;
  PetscScalar        f_lin;
  PetscReal          _f = 0.0;
  const PetscScalar *xa, *xnexta, *xenda;

  PetscFunctionBegin;
  PetscCall(VecDot(x, ctx->g_lin, &f_lin));
  _f = PetscRealPart(f_lin);
  PetscCall(AppCtxGetSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  PetscCall(QrtQuadObjective_Host(ctx, xa, xnexta, xenda, &_f));
  PetscCall(AppCtxRestoreSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  *f = _f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadGradient(Tao tao, Vec x, Vec g, void *data)
{
  AppCtx             ctx = (AppCtx)data;
  const PetscScalar *xa, *xnexta, *xenda;
  PetscScalar       *g_x, *g_y_qrt, *g_y_quad;

  PetscFunctionBegin;
  PetscCall(VecCopy(ctx->g_lin, g));
  PetscCall(AppCtxGetSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  PetscCall(AppCtxGetGradientArrays(ctx, g, &g_x, &g_y_qrt, &g_y_quad));
  PetscCall(QrtQuadGradient_Host(ctx, xa, xnexta, xenda, g_x, g_y_qrt, g_y_quad));
  PetscCall(AppCtxRestoreGradientArrays(ctx, g, &g_x, &g_y_qrt, &g_y_quad));
  PetscCall(AppCtxRestoreSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *data)
{
  AppCtx             ctx = (AppCtx)data;
  PetscReal          _f  = 0.0;
  PetscScalar        f_lin;
  const PetscScalar *xa, *xnexta, *xenda;
  PetscScalar       *g_x, *g_y_qrt, *g_y_quad;

  PetscFunctionBegin;
  PetscCall(VecDot(x, ctx->g_lin, &f_lin));
  _f = PetscRealPart(f_lin);
  PetscCall(VecCopy(ctx->g_lin, g));
  PetscCall(AppCtxGetSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  PetscCall(AppCtxGetGradientArrays(ctx, g, &g_x, &g_y_qrt, &g_y_quad));
  PetscCall(QrtQuadObjectiveGradient_Host(ctx, xa, xnexta, xenda, g_x, g_y_qrt, g_y_quad, &_f));
  PetscCall(AppCtxRestoreGradientArrays(ctx, g, &g_x, &g_y_qrt, &g_y_quad));
  PetscCall(AppCtxRestoreSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  *f = _f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QrtQuadHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *data)
{
  AppCtx             ctx = (AppCtx)data;
  const PetscScalar *xa, *xnexta, *xenda;
  PetscScalar       *h_xx, *h_xy, *h_yx, *h_yy;

  PetscFunctionBegin;
  PetscCall(AppCtxGetSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  PetscCall(AppCtxGetHessianArraysWrite(ctx, H, &h_xx, &h_xy, &h_yx, &h_yy));
  PetscCall(QrtQuadHessian_Host(ctx, xa, xnexta, xenda, h_xx, h_xy, h_yx, h_yy));
  PetscCall(AppCtxRestoreHessianArraysWrite(ctx, H, &h_xx, &h_xy, &h_yx, &h_yy));
  PetscCall(AppCtxRestoreSolutionArraysRead(ctx, x, &xa, &xnexta, &xenda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxCreateTao(AppCtx ctx, Tao *tao)
{
  Tao t;

  PetscFunctionBegin;
  PetscCall(TaoCreate(ctx->comm, tao));
  t = *tao;
  PetscCall(TaoSetSolution(t, ctx->x));
  PetscCall(TaoSetVariableBounds(t, ctx->xl, ctx->xu));
  PetscCall(TaoSetType(t, TAOBLMVM));
  PetscCall(TaoSetObjective(t, QrtQuadObjective, (void *)ctx));
  PetscCall(TaoSetGradient(t, ctx->g, QrtQuadGradient, (void *)ctx));
  PetscCall(TaoSetObjectiveAndGradient(t, ctx->g, QrtQuadObjectiveAndGradient, (void *)ctx));
  PetscCall(TaoSetHessian(t, ctx->H, ctx->H, QrtQuadHessian, (void *)ctx));
  if (ctx->set_from_options_called) PetscCall(TaoSetFromOptions(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxDestroy(AppCtx *ctx)
{
  AppCtx c;

  PetscFunctionBegin;
  c    = *ctx;
  *ctx = NULL;
  PetscCall(VecScatterDestroy(&c->g_y_qrt_scatter));
  PetscCall(VecDestroy(&c->g_y_qrt));
  PetscCall(VecDestroy(&c->g_end_local));
  PetscCall(VecDestroy(&c->g_end_ones));
  PetscCall(MatDestroy(&c->g_end_mat));
  PetscCall(VecDestroy(&c->g_end));
  PetscCall(VecDestroy(&c->g_lin));
  PetscCall(VecDestroy(&c->g));
  PetscCall(VecDestroy(&c->xu));
  PetscCall(VecDestroy(&c->xl));
  PetscCall(PetscSFDestroy(&c->x_end_bcast));
  PetscCall(VecDestroy(&c->x_end));
  PetscCall(VecScatterDestroy(&c->x_next_scatter));
  PetscCall(VecDestroy(&c->x_next));
  PetscCall(VecDestroy(&c->x));
  PetscCall(VecDestroy(&c->Hvals));
  PetscCall(MatDestroy(&c->H));
  PetscCall(PetscLayoutDestroy(&c->layout));
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx   ctx;
  MPI_Comm comm;
  Tao      tao;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(AppCtxCreate(comm, &ctx));
  PetscCall(AppCtxSetFromOptions(ctx));
  PetscCall(AppCtxSetUp(ctx));
  PetscCall(AppCtxCreateTao(ctx, &tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(AppCtxDestroy(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 1
    args: -n 50 -m 11 -k 2 -tao_type bnls

TEST*/
