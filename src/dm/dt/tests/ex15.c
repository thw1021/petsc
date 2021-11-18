const char help[] = "Test polynomial symmetrization.\n\n";

#include <petscdt.h>
#include <petscmat.h>
#include <petscblaslapack.h>

/* OEIS A000041: https://oeis.org/A000041 */
const static PetscInt PartitionFunctionValues[] = {
  1,1,2,3,5,7,11,15,22,30,42,56,77,101,135,176,231,
  297,385,490,627,792,1002,1255,1575,1958,2436,3010,
  3718,4565,5604,6842,8349,10143,12310,14883,17977,
  21637,26015,31185,37338,44583,53174,63261,75175,
  89134,105558,124754,147273,173525
};

// with a lookup table of previous computed values
static PetscInt PetscDTPartitionFunction_Single(PetscInt m, const PetscInt pt[])
{
  PetscInt p = 0;

  for (PetscInt k = 1, sign = 1; ; k++, sign *= -1) {
    PetscInt nk = -k;
    PetscInt pos_offset = (k * (3 * k - 1)) / 2;
    PetscInt neg_offset = (nk * (3 * nk - 1)) / 2;
    if (pos_offset <= m) {
      p += sign * pt[m - pos_offset];
    }
    if (neg_offset <= m) {
      p += sign * pt[m - neg_offset];
    }
    if (pos_offset > m && neg_offset > m) {
      break;
    }
  }
  return p;
}

static PetscErrorCode PetscDTPartitionFunctionUpTo_Reference(PetscInt n, PetscInt p[])
{
  PetscFunctionBegin;
  if (n < 0) {
    PetscFunctionReturn(0);
  }
  p[0] = 1;
  for (PetscInt m = 1; m <= n; m++) {
    p[m] = PetscDTPartitionFunction_Single(m, p);
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PetscDTPartitionFunctionUpTo(PetscInt n, PetscInt p[])
{
  PetscInt       n_static = sizeof(PartitionFunctionValues) / sizeof(PetscInt);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscArraycpy(p, PartitionFunctionValues, PetscMin(n+1,n_static));CHKERRQ(ierr);
  if (n >= n_static) {
    for (PetscInt m = n_static; m <= n; m++) {
      p[m] = PetscDTPartitionFunction_Single(m, p);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTPartitionFunction(PetscInt n, PetscInt *pf)
{
  PetscInt       n_static = sizeof(PartitionFunctionValues) / sizeof(PetscInt);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) {
    *pf = 0;
  } else if (n < n_static) {
    *pf = PartitionFunctionValues[n];
  } else {
    PetscInt *p;

    ierr = PetscMalloc1(n+1, &p);CHKERRQ(ierr);
    ierr = PetscDTPartitionFunctionUpTo(n, p);CHKERRQ(ierr);
    *pf = p[n];
    ierr = PetscFree(p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// ydrows and ydcols should be length partition(n) * n
static PetscErrorCode PetscDTYoungDiagrams(PetscInt n, PetscInt ydrows[], PetscInt ydcols[])
{
  PetscInt       pn;
  PetscInt       *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n <= 0) {
    PetscFunctionReturn(0);
  }
  ierr = PetscDTPartitionFunction(n, &pn);CHKERRQ(ierr);
  ierr = PetscArrayzero(ydrows, pn*n);CHKERRQ(ierr);
  ierr = PetscArrayzero(ydcols, pn*n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &x);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; i++) x[i] = 1;
  x[0] = ydrows[0] = n;
  for (PetscInt d = 1, h = 0, m = 1; d < pn; d++) {
    PetscInt *rows = &ydrows[d * n];
    if (x[h] == 2) {
      m++;
      x[h--] = 1;
    } else {
      PetscInt r = x[h] - 1;
      PetscInt t = m - h;

      x[h] = r;
      while (t >= r) {
        x[++h] = r;
        t -= r;
      }
      if (t == 0) {
        m = h+1;
      } else {
        m = h+2;
        if (t > 1) {
          x[++h] = t;
        }
      }
    }
    for (PetscInt i = 0; i < m; i++) rows[i] = x[i];
  }
  for (PetscInt d = 0; d < pn; d++) {
    const PetscInt *rows = &ydrows[d * n];
    PetscInt *cols = &ydcols[d * n];
    PetscInt filled = 0;
    for (PetscInt r = n-1; r >= 0; r--) {
      if (rows[r] > filled) {
        for (PetscInt c = filled; c < rows[r]; c++) {
          cols[c] = r+1;
        }
        filled = rows[r];
      }
    }
  }
  ierr = PetscFree(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTIrrepDim(PetscInt n, const PetscInt ydrows[], const PetscInt ydcols[], PetscInt *irrepdim)
{
  PetscInt id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTFactorialInt(n, &id);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; i++) {
    for (PetscInt j = 0; j < ydrows[i]; j++) {
      PetscInt hooklength = 1 + (ydrows[i] - 1 - j) + (ydcols[j] - 1 - i);
      id /= hooklength;
    }
  }
  *irrepdim = id;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTGetIrrepMats(PetscInt n, const PetscInt rows[], const PetscInt cols[],
                                          Mat **irrep_mats)
{
  PetscInt       dim, fact;
  PetscInt       *perm, *perm2, *perm3;
  PetscInt       *yd;
  PetscScalar    *bvec, *svec;
  PetscScalar    *sorbit;
  Mat            *im;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTIrrepDim(n, rows, cols, &dim);CHKERRQ(ierr);
  ierr = PetscDTFactorialInt(n, &fact);CHKERRQ(ierr);
  ierr = PetscMalloc1(fact, &im);CHKERRQ(ierr);
  ierr = PetscCalloc1(fact, &bvec);CHKERRQ(ierr);
  ierr = PetscCalloc1(fact, &svec);CHKERRQ(ierr);
  ierr = PetscMalloc3(n, &perm, n, &perm2, n, &perm3);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*n, &yd);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n*n; i++) yd[i] = -1;
  for (PetscInt l = 0, i = 0; i < n; i++) {
    for (PetscInt j = 0; j < rows[i]; j++) {
      yd[i*n + j] = l++;
    }
  }
  for (PetscInt p = 0; p < fact; p++) {
    PetscBool isOdd;
    PetscBool all_found;

    ierr = PetscDTEnumPerm(n, p, perm, &isOdd);CHKERRQ(ierr);
    all_found = PETSC_TRUE;
    for (PetscInt col = 0; col < n; col++) {
      for (PetscInt row = 0; row < cols[col]; row++) {
        PetscInt elem = yd[row*n + col];
        PetscInt pelem = perm[elem];
        PetscBool found = PETSC_FALSE;
        for (PetscInt row2 = 0; row2 < cols[col]; row2++) {
          if (yd[row2*n + col] == pelem) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          all_found = PETSC_FALSE;
          break;
        }
      }
      if (!all_found) {
        break;
      }
    }
    if (all_found) { // this permutation preserves columns
      bvec[p] = (isOdd == PETSC_TRUE) ? -1. : 1.;
    }
  }
  for (PetscInt p = 0; p < fact; p++) {
    PetscBool isOdd;
    PetscBool all_found;

    ierr = PetscDTEnumPerm(n, p, perm, &isOdd);CHKERRQ(ierr);
    all_found = PETSC_TRUE;
    for (PetscInt row = 0; row < n; row++) {
      for (PetscInt col = 0; col < rows[row]; col++) {
        PetscInt elem = yd[row*n + col];
        PetscInt pelem = perm[elem];
        PetscBool found = PETSC_FALSE;
        for (PetscInt col2 = 0; col2 < rows[row]; col2++) {
          if (yd[row*n + col2] == pelem) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found) {
          all_found = PETSC_FALSE;
          break;
        }
      }
      if (!all_found) {
        break;
      }
    }
    if (all_found) {
      for (PetscInt j = 0; j < fact; j++) {
        if (bvec[j] != 0) {
          PetscBool isOdd;
          PetscInt i;

          ierr = PetscDTEnumPerm(n, j, perm2, &isOdd);CHKERRQ(ierr);
          for (PetscInt k = 0; k < n; k++) {
            perm3[k] = perm[perm2[k]];
          }
          ierr = PetscDTPermIndex(n, perm3, &i, &isOdd);CHKERRQ(ierr);
          svec[i] += bvec[j];
        }
      }
    }
  }
  ierr = PetscMalloc1(fact*fact, &sorbit);CHKERRQ(ierr);
  for (PetscInt p = 0; p < fact; p++) {
    PetscBool isOdd;

    ierr = PetscDTEnumPerm(n, p, perm, &isOdd);CHKERRQ(ierr);
    for (PetscInt j = 0; j < fact; j++) {
      PetscInt i;

      ierr = PetscDTEnumPerm(n, j, perm2, &isOdd);CHKERRQ(ierr);
      for (PetscInt k = 0; k < n; k++) {
        perm3[k] = perm[perm2[k]];
      }
      ierr = PetscDTPermIndex(n, perm3, &i, &isOdd);CHKERRQ(ierr);
      sorbit[p * fact + i] = svec[j];
    }
  }
  {
    PetscBLASInt M, N, lwork, info;
    PetscScalar *tau, *work;
    PetscBLASInt *pivots;

    ierr = PetscMalloc3(fact, &tau, 4*fact, &work, fact, &pivots);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(fact,&M);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(4*fact,&lwork);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(dim,&N);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    for (PetscInt i = 0; i < fact; i++) pivots[i] = 0;
    PetscStackCallBLAS("LAPACKgeqp3",LAPACKgeqp3_(&M,&M,sorbit,&M,pivots,tau,work,&lwork,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQP3 error");
    PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&M,&N,&N,sorbit,&M,tau,work,&lwork,&info));
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");
    ierr = PetscFree3(tau, work, pivots);CHKERRQ(ierr);
  }
  for (PetscInt p = 0; p < fact; p++) {
    PetscScalar *mat;
    PetscBool isOdd;

    ierr = MatCreateSeqDense(PETSC_COMM_SELF, dim, dim, NULL, &im[p]);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(im[p], &mat);CHKERRQ(ierr);
    ierr = PetscDTEnumPerm(n, p, perm, &isOdd);CHKERRQ(ierr);
    ierr = PetscArrayzero(mat, dim*dim);CHKERRQ(ierr);
    for (PetscInt j = 0; j < fact; j++) {
      PetscInt i;

      ierr = PetscDTEnumPerm(n, j, perm2, &isOdd);CHKERRQ(ierr);
      for (PetscInt k = 0; k < n; k++) {
        perm3[k] = perm[perm2[k]];
      }
      ierr = PetscDTPermIndex(n, perm3, &i, &isOdd);CHKERRQ(ierr);
      for (PetscInt qj = 0; qj < dim; qj++) {
        for (PetscInt qi = 0; qi < dim; qi++) {
          mat[qi + qj*dim] += sorbit[qi * fact + i] * sorbit[qj * fact + j];
        }
      }
    }
    ierr = MatDenseRestoreArrayWrite(im[p], &mat);CHKERRQ(ierr);
  }
  ierr = PetscFree(sorbit);CHKERRQ(ierr);
  ierr = PetscFree(yd);CHKERRQ(ierr);
  ierr = PetscFree3(perm, perm2, perm3);CHKERRQ(ierr);
  ierr = PetscFree(svec);CHKERRQ(ierr);
  ierr = PetscFree(bvec);CHKERRQ(ierr);
  *irrep_mats = im;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDTRestoreIrrepMats(PetscInt n, const PetscInt rows[], const PetscInt cols[],
                                              Mat **irrep_mats)
{
  Mat            *im = *irrep_mats;
  PetscInt       fact;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTFactorialInt(n, &fact);CHKERRQ(ierr);
  for (PetscInt i = 0; i < fact; i++) {
    ierr = MatDestroy(&im[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(im);CHKERRQ(ierr);
  *irrep_mats = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode testIrrepMats(PetscInt n, Mat *irrepmats)
{
  PetscInt       *permi, *permj, *permk, fact;
  Mat             outmat = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDTFactorialInt(n, &fact);CHKERRQ(ierr);
  ierr = PetscMalloc3(fact, &permi, fact, &permj, fact, &permk);CHKERRQ(ierr);
  if (fact) {
    ierr = MatDuplicate(irrepmats[0], MAT_DO_NOT_COPY_VALUES, &outmat);CHKERRQ(ierr);
  }
  for (PetscInt i = 0; i < fact; i++) {
    Mat mati = irrepmats[i];
    PetscBool isOdd;
    PetscReal errnorm;

    ierr = PetscDTEnumPerm(n, i, permi, &isOdd);CHKERRQ(ierr);
    for (PetscInt j = 0; j < fact; j++) {
      Mat matj = irrepmats[j];
      PetscInt k;

      ierr = MatMatMult(matj, mati, MAT_REUSE_MATRIX, PETSC_DEFAULT, &outmat);CHKERRQ(ierr);
      ierr = PetscDTEnumPerm(n, j, permj, &isOdd);CHKERRQ(ierr);
      for (PetscInt l = 0; l < n; l++) {
        permk[l] = permj[permi[l]];
      }
      ierr = PetscDTPermIndex(n, permk, &k, &isOdd);CHKERRQ(ierr);
      ierr = MatAXPY(outmat, -1., irrepmats[k], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(outmat, NORM_FROBENIUS, &errnorm);CHKERRQ(ierr);
      if (errnorm > PETSC_SMALL) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%D * %D = %D error, %g\n", i, j, k, errnorm);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatDestroy(&outmat);CHKERRQ(ierr);
  ierr = PetscFree3(permi, permj, permk);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode testPetscDTYoungDiagrams(PetscInt n)
{
  PetscInt       p, *ydrows, *ydcols;
  PetscInt       fact, id2sum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTPartitionFunction(n, &p);CHKERRQ(ierr);
  ierr = PetscMalloc2(p * n, &ydrows, p * n, &ydcols);CHKERRQ(ierr);
  ierr = PetscDTYoungDiagrams(n, ydrows, ydcols);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "S%D Young diagrams\n\n", n);CHKERRQ(ierr);
  ierr = PetscDTFactorialInt(n, &fact);CHKERRQ(ierr);
  id2sum = 0;
  for (PetscInt i = 0; i < p; i++) {
    const PetscInt *rows = &ydrows[i*n];
    const PetscInt *cols = &ydcols[i*n];
    PetscInt irrepdim;
    Mat *irrepmats;

    for (PetscInt j = 0; j < n; j++) {
      if (rows[j]) {
        if (j > 0) {
          ierr = PetscPrintf(PETSC_COMM_WORLD, " %D\n", rows[j]);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_WORLD, " %D \\", rows[j]);CHKERRQ(ierr);
          for (PetscInt k = 0; k < n; k++) {
            if (cols[k]) {
              ierr = PetscPrintf(PETSC_COMM_WORLD, " %D", cols[k]);CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscDTIrrepDim(n, rows, cols, &irrepdim);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n  Irrep dim: %D\n", irrepdim);CHKERRQ(ierr);
    id2sum += irrepdim*irrepdim;
    ierr = PetscDTGetIrrepMats(n, rows, cols, &irrepmats);CHKERRQ(ierr);
    ierr = testIrrepMats(n, irrepmats);CHKERRQ(ierr);
    ierr = PetscDTRestoreIrrepMats(n, rows, cols, &irrepmats);CHKERRQ(ierr);
    ierr = PetscFree(irrepmats);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }
  if (id2sum != fact) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "S%D sum of irrep dimensions %D != %D\n", n, id2sum, fact);
  ierr = PetscFree2(ydrows, ydcols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscInt       p[122];
  PetscInt       p_ref[122];
  PetscInt       p_121 = 2056148051;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscDTPartitionFunctionUpTo(121, p);CHKERRQ(ierr);
  ierr = PetscDTPartitionFunctionUpTo_Reference(121, p_ref);CHKERRQ(ierr);
  for (PetscInt i = 0; i < 121; i++) {
    PetscInt pi;

    ierr = PetscDTPartitionFunction(i, &pi);CHKERRQ(ierr);
    if (pi != p_ref[i]) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "p(%D) = %D != %D\n", i, pi, p_ref[i] );
    if (p[i] != p_ref[i]) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "p(%D) = %D != %D\n", i, p[i], p_ref[i] );
  }
  if (p[121] != p_121) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "p(%D) = %D != %D\n", 121, p[121], p_121 );
  for (PetscInt i = 0; i < 6; i++) {
    ierr = testPetscDTYoungDiagrams(i);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    args:

TEST*/
