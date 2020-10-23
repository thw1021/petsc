#if !defined(EX100_H)
#define EX100_H

#include <petscmat.h>
#include <petsc/private/matimpl.h>

#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv)    \
  {                                                                    \
  if (col <= lastcol1)  low1 = 0;                                      \
  else                 high1 = nrow1;                                  \
  lastcol1 = col;                                                      \
  while (high1-low1 > 5) {                                             \
    t = (low1+high1)/2;                                                \
    if (rp1[t] > col) high1 = t;                                       \
    else              low1  = t;                                       \
  }                                                                    \
  for (_i=low1; _i<high1; _i++) {                                      \
    if (rp1[_i] > col) break;                                          \
    if (rp1[_i] == col) {                                              \
      if (addv == ADD_VALUES) {                                        \
        Kokkos::atomic_add(&ap1[_i],value);                                     \
      }                                                                \
      else ap1[_i] = value;                                            \
      inserted = 1; Kokkos::atomic_add(&d_mat->nonzerostate,1);  \
      break;                                                           \
    }                                                                  \
  }                                                                    \
}

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv)    \
  {                                                                    \
  if (col <= lastcol2) low2 = 0;                                       \
  else high2 = nrow2;                                                  \
  lastcol2 = col;                                                      \
  while (high2-low2 > 5) {                                             \
    t = (low2+high2)/2;                                                \
    if (rp2[t] > col) high2 = t;                                       \
    else              low2  = t;                                       \
  }                                                                    \
  for (_i=low2; _i<high2; _i++) {                                      \
    if (rp2[_i] > col) break;                                          \
    if (rp2[_i] == col) {                                              \
      if (addv == ADD_VALUES) {                                        \
        Kokkos::atomic_add(&ap2[_i],value);                                     \
      }                                                                \
      else ap2[_i] = value;                                            \
      inserted = 1; Kokkos::atomic_add(&d_mat->nonzerostate,1);  \
      break;                                                           \
    }                                                                  \
  }                                                                    \
}

void MatSetValuesKokkos(PetscSplitCSRDataStructure *d_mat, PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  printf("%d %d %d %d %d %d %g %g %g\n",m,im[0],n,in[0],in[1],in[2],v[0],v[1],v[2]);
  if (m==0) return;
  else {
    MatScalar value=0.0;
    PetscInt  *ai = d_mat->diag.i;
    PetscInt  *aj = d_mat->diag.j;
    PetscBool ignorezeroentries = (d_mat->diag.ignorezeroentries==0) ? PETSC_FALSE : PETSC_TRUE;
    PetscInt  *bi = d_mat->offdiag.i, *bj = d_mat->offdiag.j;
    MatScalar *ba = d_mat->offdiag.a, *aa = d_mat->diag.a;
    PetscInt  *rp1,*rp2=NULL,nrow1,nrow2,_i,low1,high1,low2,high2,t,lastcol1,lastcol2,inserted;
    MatScalar *ap1,*ap2=NULL;
    PetscBool roworiented = PETSC_TRUE;
    PetscInt  i,j,rstart  = d_mat->rstart,rend = d_mat->rend;
    PetscInt  cstart      = d_mat->rstart,cend = d_mat->rend,row,col;

    for (i=0; i<m; i++) {
      if (im[i] >= rstart && im[i] < rend) { // ignore off processor rows
        row      = im[i] - rstart;
        lastcol1 = -1;
        rp1      = aj + ai[row];
        ap1      = aa + ai[row];
        nrow1    = ai[row+1] - ai[row];
        low1     = 0;
        high1    = nrow1;
        if (bj) {
          lastcol2 = -1;
          rp2      = bj + bi[row];
          ap2      = ba + bi[row];
          nrow2    = bi[row+1] - bi[row];
          low2     = 0;
          high2    = nrow2;
        }
        for (j=0; j<n; j++) {
          if (v)  value = roworiented ? v[i*n+j] : v[i+j*m];
          if (ignorezeroentries && value == 0.0 && (is == ADD_VALUES) && im[i] != in[j]) continue;
          if (in[j] >= cstart && in[j] < cend) {
            col   = in[j] - cstart;
            inserted = 0;
            MatSetValues_SeqAIJ_A_Private(row,col,value,is);

          } else if (in[j] < 0) {
            continue; // need to checm for > N also
          } else {
            col = d_mat->colmap[in[j]] - 1;
            inserted = 0;
            MatSetValues_SeqAIJ_B_Private(row,col,value,is);
          }
        }
      }
    }
  }
}
#endif
