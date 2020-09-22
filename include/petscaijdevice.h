#if !defined(__PETSCAIJDEVICE_H)
#define __PETSCAIJDEVICE_H

#include <petscmat.h>
#include <petsc/private/matimpl.h>

#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv,orow,ocol)     \
  {                                                                     \
  if (col <= lastcol1)  low1 = 0;                                       \
  else                 high1 = nrow1;                                   \
  lastcol1 = col;                                                       \
  while (high1-low1 > 5) {                                              \
    t = (low1+high1)/2;                                                 \
    if (rp1[t] > col) high1 = t;                                        \
    else              low1  = t;                                        \
  }                                                                     \
  for (_i=low1; _i<high1; _i++) {                                       \
    if (rp1[_i] > col) break;                                           \
    if (rp1[_i] == col) {                                               \
      if (addv == ADD_VALUES) {                                         \
        ap1[_i] += value;                                               \
      }                                                                 \
      else         ap1[_i] = value;                                     \
      /*inserted = PETSC_TRUE;*/                                        \
      break;                                                            \
    }                                                                   \
  }                                                                     \
  if (value == 0.0 && ignorezeroentries && row != col) {low1 = 0; high1 = nrow1;} \
  else if (nonew == 1) {low1 = 0; high1 = nrow1;}                       \
  else if (nonew == -1) {                                               \
    printf("A: Inserting a new nonzero at global row/column (%d, %d) into matrix\n", orow, ocol); \
    *ierr = 1;                                                          \
  } else if (nrow1 >= rmax1) {                                          \
    printf("A: ERROR, ran out of preallocated space in row %d\n", orow);\
    *ierr = 2;                                                          \
  } else {                                                              \
    printf("A: column %d not found in row %d\n", col, orow);            \
    *ierr = 3;                                                          \
  }                                                                     \
}

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv,orow,ocol)     \
  {                                                                     \
  if (col <= lastcol2) low2 = 0;                                        \
  else high2 = nrow2;                                                   \
  lastcol2 = col;                                                       \
  while (high2-low2 > 5) {                                              \
    t = (low2+high2)/2;                                                 \
    if (rp2[t] > col) high2 = t;                                        \
    else             low2  = t;                                         \
  }                                                                     \
  for (_i=low2; _i<high2; _i++) {                                       \
    if (rp2[_i] > col) break;                                           \
    if (rp2[_i] == col) {                                               \
      if (addv == ADD_VALUES) {                                         \
        ap2[_i] += value;                                               \
      }                                                                 \
      else                    ap2[_i] = value;                          \
      /*inserted = PETSC_TRUE;*/                                        \
      break;                                                            \
    }                                                                   \
  }                                                                     \
  if (value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2; }    \
  else if (nonew == 1) {low2 = 0; high2 = nrow2; }                      \
  else if (nonew == -1) {                                                    \
    printf("B: Inserting a new nonzero at global row/column (%d, %d) into matrix\n", orow, ocol); \
    *ierr = 1;                                                          \
  } else if (nrow2 >= rmax2) {                                          \
    printf("B: ERROR, ran out of preallocated space in row %d\n", orow);\
    *ierr = 2;                                                          \
  } else {                                                              \
    printf("B: column %d not found in row %d\n", col, orow);  \
    *ierr = 3;                                                          \
  }                                                                     \
}

#if defined(PETSC_HAVE_CUDA)
  static __device__
#endif // PETSC_HAVE_CUDA
void MatSetValues_AIJ_device(PetscSplitCSRDataStructure *d_mat, PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is, PetscErrorCode *ierr)
{
  MatScalar value=0.0;
  PetscInt  *aimax = d_mat->diag.imax,*ai = d_mat->diag.i,*ailen = d_mat->diag.ilen;
  PetscInt  *aj = d_mat->diag.j, nonew = d_mat->diag.nonew; // same for A and B
  PetscBool ignorezeroentries = (d_mat->diag.ignorezeroentries==0) ? PETSC_FALSE : PETSC_TRUE;
  PetscInt  *bimax = d_mat->offdiag.imax,*bi = d_mat->offdiag.i, *bilen = d_mat->offdiag.ilen, *bj = d_mat->offdiag.j;
  MatScalar *ba = d_mat->offdiag.a, *aa = d_mat->diag.a;
  PetscInt  *rp1,*rp2=NULL,nrow1,nrow2,_i,rmax1,rmax2,N,low1,high1,low2,high2,t,lastcol1,lastcol2;
  MatScalar *ap1,*ap2=NULL;
  PetscBool roworiented = PETSC_TRUE;
  PetscInt  i,j,rstart  = d_mat->rstart,rend = d_mat->rend;
  PetscInt  cstart      = d_mat->rstart,cend = d_mat->rend,row,col;
  //PetscBool inserted = PETSC_FALSE;

  *ierr = 0;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    //if (PetscUnlikelyDebug(im[i] >= mat->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %d max %d",im[i],mat->rmap->N-1);
    if (im[i] >= rstart && im[i] < rend) {
      row      = im[i] - rstart;
      lastcol1 = -1;
      rp1      = aj + ai[row];
      ap1      = aa + ai[row];
      rmax1    = aimax[row];
      nrow1    = ailen[row];
      low1     = 0;
      high1    = nrow1;
      if (bj) {
        lastcol2 = -1;
        rp2      = bj + bi[row];
        ap2      = ba + bi[row];
        rmax2    = bimax[row];
        nrow2    = bilen[row];
        low2     = 0;
        high2    = nrow2;
      }
      for (j=0; j<n; j++) {
        if (v)  value = roworiented ? v[i*n+j] : v[i+j*m];
        if (ignorezeroentries && value == 0.0 && (is == ADD_VALUES) && im[i] != in[j]) continue;
        if (in[j] >= cstart && in[j] < cend) {
          col   = in[j] - cstart;
          MatSetValues_SeqAIJ_A_Private(row,col,value,is,im[i],in[j]);
          if (*ierr) return;
          //if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) A->offloadmask = PETSC_OFFLOAD_CPU;
        } else if (in[j] < 0) {
          continue;
          // else if (PetscUnlikelyDebug(in[j] >= mat->cmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap->N-1);
        } else {
          // if (mat->was_assembled)
          if (!d_mat->colmap) {
            printf("ERROR, !d_mat->colmap\n");
            return;
          }
          #if defined(PETSC_USE_CTABLE)
          #error
          PetscTableFind(aij->colmap,in[j]+1,&col); // todo
          col--;
          #else
          col = d_mat->colmap[in[j]] - 1;
          #endif
          if (col < 0) printf("ERROR col %d not found (%d)\n",in[j],col);
          MatSetValues_SeqAIJ_B_Private(row,col,value,is,im[i],in[j]);
          if (*ierr) return;
          //if (B->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) B->offloadmask = PETSC_OFFLOAD_CPU;
        }
      }
    } else {
      printf("ERROR, off processor rows not supported. No stash. row %d\n",(int)im[i]);
    }
  }
  //if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) A->offloadmask = PETSC_OFFLOAD_CPU;
}

#endif // __PETSCAIJDEVICE_H
