#include <petscsys.h>                /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>

PETSC_STATIC_INLINE int Compare_PetscMPIInt_Private(const void *left, const void *right)
{
  const PetscMPIInt l = *(const PetscMPIInt *) left, r = *(const PetscMPIInt *) right;
  return l < r ? -1 : l == r ? 0 : 1;
}

PETSC_STATIC_INLINE int Compare_PetscInt_Private(const void *left, const void *right)
{
  const PetscInt l = *(const PetscInt *) left, r = *(const PetscInt *) right;
  return l < r ? -1 : l == r ? 0 : 1;
}

#define SMALL_STACK_SIZE 512
#define MIN_GALLOP_CONST_GLOBAL 8
static PetscInt MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;
typedef int (*CompFunc)(const void *, const void *);
typedef unsigned int uPetscInt;

#define _memcpy(a,b,c)  __builtin_memcpy((char *)(a),(char *)(b),(size_t)(c))
#define _memmove(a,b,c) __builtin_memmove((char *)(a),(char *)(b),(size_t)(c))
#define COPYSWAP(a,b,t,size)                                                \
  do {                                                                  \
    _memcpy((t),(b),(size));                                            \
    _memmove((b),(a),(size));                                           \
    _memcpy((a),(t),(size));                                            \
  } while (0)

/* Start left look right. Looking for e.g. B[0] in A or mergelo. l inclusive, r inclusive. Returns first m such that arr[m] >
 x. Output also inclusive */
PETSC_STATIC_INLINE PetscErrorCode PetscGallopSearchLeft_Private(const void *arr, size_t size, CompFunc cmp, PetscInt l, PetscInt r, const void *x, PetscInt *m)
{
  PetscInt last = l, k = 1, mid, cur = l+1;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(r < l)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"r %D < l %D in PetscGallopSearchLeft",r,l);
  if (PetscUnlikely(!r-l)) {*m = l;PetscFunctionReturn(0);}
  if ((*cmp)(x, (char *)(arr)+l*size) < 0) {*m = l;PetscFunctionReturn(0);}
  while (PETSC_TRUE) {
    if (cur > r) {cur = r; break;}
    if ((*cmp)(x, (char *)(arr)+cur*size) < 0) break;
    last = cur;
    cur += (k <<= 1) + 1; ++k;
  }
  /* standard binary search but take last 0 mid 0 cur 1 into account*/
  while (cur > last + 1) {
    mid = last + ((cur - last) >> 1);
    if ((*cmp)(x, (char *)(arr)+mid*size) < 0) {
      cur = mid;
    } else {
      last = mid;
    }
  }
  *m = cur;
  PetscFunctionReturn(0);
}

/* Start right look left. Looking for e.g. B[-1] in A or mergehi. l inclusive, r inclusive. Returns last m such that arr[m]
 < x. Output also inclusive */
PETSC_STATIC_INLINE PetscErrorCode PetscGallopSearchRight_Private(const void *arr, size_t size, CompFunc cmp, PetscInt l, PetscInt r, const void *x, PetscInt *m)
{
  PetscInt last = r, k = 1, mid, cur = r-1;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(r < l)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"r %D < l %D in PetscGallopSearchRight",r,l);
  if (PetscUnlikely(!r-l)) {*m = r;PetscFunctionReturn(0);}
  if ((*cmp)(x, (char *)(arr)+r*size) > 0) {*m = r;PetscFunctionReturn(0);}
  while (PETSC_TRUE) {
    if (cur < l) {cur = l; break;}
    if ((*cmp)(x, (char *)(arr)+cur*size) > 0) break;
    last = cur;
    cur -= (k <<= 1) + 1; ++k;
  }
  /* standard binary search but take last r-1 mid r-1 cur r-2 into account*/
  while (last > cur + 1) {
    mid = last - ((last - cur) >> 1);
    if ((*cmp)(x, (char *)(arr)+mid*size) > 0) {
      cur = mid;
    } else {
      last = mid;
    }
  }
  *m = cur;
  PetscFunctionReturn(0);
}

/* Mergesort where size of left half <= size of right half, so mergesort is done left to right. Arr should be pointer to
 complete array, left is first index of left array, mid is first index of right array, right is last index of right
 array */
PETSC_STATIC_INLINE PetscErrorCode PetscTimSortMergeLo_Private(void *arr, void *tarr, size_t size, CompFunc cmp, PetscInt left, PetscInt mid, PetscInt right)
{
  PetscInt       i = 0, j = mid, k = left, gallopleft = 0, gallopright = 0;
  const PetscInt llen = mid-left;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  _memcpy(tarr, (char *)(arr)+left*size, llen*size);
  while ((i < llen) && (j <= right)) {
    if ((*cmp)((char *)(tarr)+i*size, (char *)(arr)+j*size) < 0) {
      _memcpy((char *)(arr)+k*size, (char *)(tarr)+i*size, size);
      ++k; ++i;
      gallopright = 0;
      if (++gallopleft >= MIN_GALLOP_GLOBAL && i < llen) {
        PetscInt l1, l2, diff1, diff2;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search temp for right[j], can move up to that of temp into arr immediately */
          ierr = PetscGallopSearchLeft_Private(tarr, size, cmp, i, llen-1, (char *)(arr)+j*size, &l1);CHKERRQ(ierr);
          diff1 = l1-i;
          _memcpy((char *)(arr)+k*size, (char *)(tarr)+i*size, diff1*size);
          k += diff1;
          i = l1;
          /* search right for temp[i], can move up to that many of right into arr */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, j, right, (char *)(tarr)+i*size, &l2);CHKERRQ(ierr);
          diff2 = l2-j;
          _memmove((char *)(arr)+k*size, (char *)(arr)+j*size, diff2*size);
          k += diff2;
          j = l2;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    } else {
      _memmove((char *)(arr)+k*size, (char *)(arr)+j*size, size);
      ++k; ++j;
      gallopleft = 0;
      if (++gallopright >= MIN_GALLOP_GLOBAL && j <= right) {
        PetscInt l1, l2, diff1, diff2;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search right for temp[i], can move up to that many of right into arr */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, j, right, (char *)(tarr)+i*size, &l2);CHKERRQ(ierr);
          diff2 = l2-j;
          _memmove((char *)(arr)+k*size, (char *)(arr)+j*size, diff2*size);
          k += diff2;
          j = l2;
          /* search temp for right[j], can copy up to that of temp into arr immediately */
          ierr = PetscGallopSearchLeft_Private(tarr, size, cmp, i, llen-1, (char *)(arr+j*size), &l1);CHKERRQ(ierr);
          diff1 = l1-i;
          _memcpy((char *)(arr)+k*size, (char *)(tarr)+i*size, diff1*size);
          k += diff1;
          i = l1;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff1 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    }
  }
  if (i<llen) {_memcpy((char *)(arr)+k*size, (char *)(tarr)+i*size, (llen-i)*size);}
  PetscFunctionReturn(0);
}

/* Mergesort where size of right half < size of left half, so mergesort is done right to left. Arr should be pointer to
 complete array, left is first index of left array, mid is first index of right array, right is last index of right
 array */
PETSC_STATIC_INLINE PetscErrorCode PetscTimSortMergeHi_Private(void *arr, void *tarr, size_t size, CompFunc cmp, PetscInt left, PetscInt mid, PetscInt right)
{
  PetscInt       i = right-mid, j = mid-1, k = right, gallopleft = 0, gallopright = 0;
  const PetscInt rlen = right-mid+1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  _memcpy(tarr, (char *)(arr)+mid*size, rlen*size);
  while ((i >= 0) && (j >= left)) {
    if ((*cmp)((char *)(tarr)+i*size, (char *)(arr)+j*size) > 0) {
      _memcpy((char *)(arr)+k*size, (char *)(tarr)+i*size, size);
      --k; --i;
      gallopleft = 0;
      if (++gallopright >= MIN_GALLOP_GLOBAL && i >= 0) {
        PetscInt l1, l2, diff1, diff2;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search temp for left[j], can copy up to that many of temp into arr */
          ierr = PetscGallopSearchRight_Private(tarr, size, cmp, 0, i, (char *)(arr)+j*size, &l1);CHKERRQ(ierr);
          diff1 = i-l1;
          _memcpy((char *)(arr)+(k-diff1+1)*size, (char *)(tarr)+(l1+1)*size, diff1*size);
          k -= diff1;
          i = l1;
          /* search left for temp[i], can move up to that many of left up arr */
          ierr = PetscGallopSearchRight_Private(arr, size, cmp, left, j, (char *)(tarr)+i*size, &l2);CHKERRQ(ierr);
          diff2 = j-l2;
          _memmove((char *)(arr)+(k-diff2+1)*size, (char *)(arr)+(l2+1)*size, diff2*size);
          k -= diff2;
          j = l2;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    } else {
      _memmove((char *)(arr)+k*size, (char *)(arr)+j*size, size);
      --k; --j;
      gallopright = 0;
      if (++gallopleft >= MIN_GALLOP_GLOBAL && j >= left) {
        PetscInt l1, l2, diff1, diff2;
        do {
          if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;
          /* search left for temp[i], can move up to that many of left up arr */
          ierr = PetscGallopSearchRight_Private(arr, size, cmp, left, j, (char *)(tarr)+i*size, &l2);CHKERRQ(ierr);
          diff2 = j-l2;
          _memmove((char *)(arr)+(k-diff2+1)*size, (char *)(arr)+(l2+1)*size, diff2*size);
          k -= diff2;
          j = l2;
          /* search temp for left[j], can copy up to that many of temp into arr */
          ierr = PetscGallopSearchRight_Private(tarr, size, cmp, 0, i, (char *)(arr)+j*size, &l1);CHKERRQ(ierr);
          diff1 = i-l1;
          _memcpy((char *)(arr)+(k-diff1+1)*size, (char *)(tarr)+(l1+1)*size, diff1*size);
          k -= diff1;
          i = l1;
        } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL);
        ++MIN_GALLOP_GLOBAL;
      }
    }
  }
  if (i >= 0) {_memcpy((char *)(arr)+left*size, tarr, (i+1)*size);}
  PetscFunctionReturn(0);
}

/* Left is inclusive lower bound of array slice, start is start location of unsorted section, right is inclusive upper
 bound of array slice. If unsure of where unsorted section starts or if entire length is unsorted pass start = left */
PETSC_STATIC_INLINE PetscErrorCode PetscInsertionSort_Private(void *arr, void *tarr, size_t size, CompFunc cmp, PetscInt left, PetscInt start, PetscInt right)
{
  PetscInt i;

  PetscFunctionBegin;
  if (start == left) ++start;
  for (i = start; i <= right; ++i) {
    PetscInt j = i-1;
    _memcpy(tarr, (char *)(arr)+i*size, size);
    while ((j >= left) && ((*cmp)(tarr, (char *)(arr)+j*size) < 0)) {
      COPYSWAP((char *)(arr)+(j+1)*size,(char *)(arr)+j*size,(char *)(tarr)+size,size);
      --j;
    }
    _memcpy((char *)(arr)+(j+1)*size, tarr, size);
  }
  PetscFunctionReturn(0);
}

/* See PetscInsertionSort_Private */
PETSC_STATIC_INLINE PetscErrorCode PetscBinaryInsertionSort_Private(void *arr, void *tarr, size_t size, CompFunc cmp, PetscInt left, PetscInt start, PetscInt right)
{
  PetscInt i;

  PetscFunctionBegin;
  if (start == left) ++start;
  for (i = start; i <= right; ++i) {
    PetscInt l = left, r = i;
    _memcpy(tarr, (char *)(arr)+i*size, size);
    do {
      PetscInt m;
      m = l + ((r - l) >> 1);
      if ((*cmp)(tarr, (char *)(arr)+m*size) < 0) {
        r = m;
      } else {
        l = m + 1;
      }
    } while (l < r);
    _memmove((char *)(arr)+(l+1)*size, (char *)(arr)+l*size, (i-l)*size);
    _memcpy((char *)(arr)+l*size, tarr, size);
  }
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt size;
  PetscInt start;
} PetscTimSortStack PETSC_ATTRIBUTEALIGNED(PETSC_MEMALIGN);

typedef struct {
  char     *ptr;
  PetscInt size;
  PetscInt maxsize;
} PetscTimSortBuffer PETSC_ATTRIBUTEALIGNED(PETSC_MEMALIGN);

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortResizeBuffer_Private(PetscTimSortBuffer *buff, size_t newSize)
{
  PetscFunctionBegin;
  if (PetscLikely(newSize <= buff->size)) PetscFunctionReturn(0);
  {
    /* Can't be larger than n, there is merit to simply allocating buff to n to begin with */
    PetscErrorCode ierr;
    size_t         newMax = PetscMin(newSize*newSize, buff->maxsize);
    ierr = PetscFree(buff->ptr);CHKERRQ(ierr);
    ierr = PetscMalloc1(newMax, &buff->ptr);CHKERRQ(ierr);
    buff->size = newMax;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortForceCollapse_Private(void *arr, size_t size, CompFunc cmp, PetscTimSortBuffer *buff, PetscTimSortStack *stack, PetscInt stacksize)
{
  PetscFunctionBegin;
  while (stacksize) {
    PetscInt       l, m = stack[stacksize].start, r;
    PetscErrorCode ierr;

    /* A = stack[i-1], B = stack[i] */
    /* Search A for B[0] insertion */
    ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[stacksize-1].start, stack[stacksize].start-1, (char *)(arr)+(stack[stacksize].start)*size, &l);CHKERRQ(ierr);
    /* l == m-1 means sorted */
    if (l < m-1) {
      /* Search B for A[-1] insertion */
      ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[stacksize].start, stack[stacksize].start+stack[stacksize].size-1, (char *)(arr)+(stack[stacksize].start-1)*size, &r);CHKERRQ(ierr);
      if (m-l <= r-m) {
        ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*size);CHKERRQ(ierr);
        ierr = PetscTimSortMergeLo_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
      } else {
        ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*size);CHKERRQ(ierr);
        ierr = PetscTimSortMergeHi_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
      }
    }
    /* Update A with merge */
    stack[stacksize-1].size += stack[stacksize].size;
    --stacksize;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortMergeCollapse_Private(void *arr, size_t size, CompFunc cmp, PetscTimSortBuffer *buff, PetscTimSortStack *stack, PetscInt *stacksize)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  i = *stacksize;
  while (i) {
    PetscInt l, m, r, itemp = i;

    if (i == 1) {
      /* A = stack[i-1], B = stack[i] */
      if (stack[i-1].size < stack[i].size) {
        m = stack[i].start;
        /* Search A for B[0] insertion */
        ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[i-1].start, stack[i].start-1, (char *)(arr)+stack[i].start*size, &l);CHKERRQ(ierr);
        /* l == m-1 means sorted */
        if (l < m-1) {
          /* Search B for A[-1] insertion */
          ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[i].start, stack[i].start+stack[i].size-1, (char *)(arr)+(stack[i].start-1)*size, &r);CHKERRQ(ierr);
          if (m-l <= r-m) {
            ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*size);CHKERRQ(ierr);
            ierr = PetscTimSortMergeLo_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
          } else {
            ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*size);CHKERRQ(ierr);
            ierr = PetscTimSortMergeHi_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
          }
        }
        /* Update A with merge */
        stack[i-1].size += stack[i].size;
        --i;
      }
    } else {
      /* i > 2, i.e. C exists
       A = stack[i-2], B = stack[i-1], C = stack[i]; */
      if (stack[i-2].size <= stack[i-1].size+stack[i].size) {
        if (stack[i-2].size < stack[i].size) {
          /* merge B into A */
          m = stack[i-1].start;
          /* Search A for B[0] insertion */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[i-2].start, stack[i-1].start-1, (char *)(arr)+(stack[i-1].start)*size, &l);CHKERRQ(ierr);
          if (l < m-1) {
            /* Search B for A[-1] insertion */
            ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[i-1].start, stack[i-1].start+stack[i-1].size-1, (char *)(arr)+(stack[i-1].start-1)*size, &r);CHKERRQ(ierr);
            if (m-l <= r-m) {
              ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*size);CHKERRQ(ierr);
              ierr = PetscTimSortMergeLo_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
            } else {
              ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*size);CHKERRQ(ierr);
              ierr = PetscTimSortMergeHi_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
            }
          }
          /* Update A with merge */
          stack[i-2].size += stack[i-1].size;
          /* Push C up the stack */
          stack[i-1].start = stack[i].start;
          stack[i-1].size = stack[i].size;
        } else {
          /* merge C into B */
          mergeBC:
          m = stack[i].start;
          /* Search B for C[0] insertion */
          ierr = PetscGallopSearchLeft_Private(arr, size, cmp, stack[i-1].start, stack[i].start-1, (char *)(arr)+stack[i].start*size, &l);CHKERRQ(ierr);
          if (l < m-1) {
            /* Search C for B[-1] insertion */
            ierr = PetscGallopSearchRight_Private(arr, size, cmp, stack[i].start, stack[i].start+stack[i].size-1, (char *)(arr)+(stack[i].start-1)*size, &r);CHKERRQ(ierr);
            if (m-l <= r-m) {
              ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*size);CHKERRQ(ierr);
              ierr = PetscTimSortMergeLo_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
            } else {
              ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*size);CHKERRQ(ierr);
              ierr = PetscTimSortMergeHi_Private(arr, buff->ptr, size, cmp, l, m, r);CHKERRQ(ierr);
            }
          }
          /* Update B with merge */
          stack[i-1].size += stack[i].size;
        }
        --i;
      } else if (stack[i-1].size <= stack[i].size) {
        /* merge C into B */
        goto mergeBC;
      }
    }
    if (itemp == i) break;
  }
  *stacksize = i;
  PetscFunctionReturn(0);
}

/* March sequentially through the array building up a "run" of weakly increasing or strictly decreasing contiguous
 elements. Decreasing runs are reversed by swapping. If the run is less than minrun, artificially extend it via either
 binary insertion sort or regulat insertion sort */
PETSC_STATIC_INLINE PetscErrorCode PetscTimSortBuildRun_Private(PetscInt n, void *arr, void *tarr, size_t size, CompFunc cmp, PetscInt minrun, PetscInt runstart, PetscInt *runend)
{
  const PetscInt re = PetscMin(runstart+minrun, n-1);
  PetscInt       ri = runstart;

  PetscFunctionBegin;
  if (PetscUnlikely(runstart == n-1)) {*runend = runstart; PetscFunctionReturn(0);}
  /* guess whether run is ascending or descending and tally up the longest consecutive run. essentially a coinflip for random data */
  if ((*cmp)((char *)(arr)+(ri+1)*size, (char *)(arr)+ri*size) < 0) {
    ++ri;
    while (ri < n-1) {
      if ((*cmp)((char *)(arr)+(ri+1)*size, (char *)(arr)+ri*size) >= 0) break;
      ++ri;
    }
    {
      PetscInt lo = runstart, hi = ri;
      do{
        COPYSWAP((char *)(arr)+lo*size,(char *)(arr)+hi*size,tarr,size);
        ++lo; --hi;
      } while (lo < hi);
    }
  } else {
    ++ri;
    while (ri < n-1) {
      if ((*cmp)((char *)(arr)+(ri+1)*size, (char *)(arr)+ri*size) < 0) break;
      ++ri;
    }
  }
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    ierr = PetscInfo1(NULL, "natural run length = %D\n", ri-runstart+1);CHKERRQ(ierr);
  }
#endif
  if (ri < re) {
    /* the attempt failed, this section likely contains random data. If ri got close to minrun (within 50%) then we try
     binary search */
    if (ri-runstart <= minrun >> 1) {
      ++MIN_GALLOP_GLOBAL; /* didn't get close hedge our bets against random data */
      PetscInsertionSort_Private(arr, tarr, size, cmp, runstart, ri, re);
    } else {
      PetscBinaryInsertionSort_Private(arr, tarr, size, cmp, runstart, ri, re);
    }
    *runend = re;
  } else *runend = ri;
  PetscFunctionReturn(0);
}

/*
  PetscTimSort - Sorts an array in place in increasing order using Tim Peters adaptive sorting algorithm.

  Not Collective

  Input Parameters:
+ n    - number of values
. arr  - array to be sorted
. size - size in bytes of the datatype held in arr
- cmp  - function pointer to comparison function

  Output Parameters:
. arr  - sorted array

  Sample usage:
  The comparison function should take a left and right argument and return the signed difference between the two. The
 contents of the void pointers should be cast to the correct type inside the comparison function. For example when
 sorting an array of type "my_type" in increasing order.
.vb
  PetscInt my_increasing_comparison_function(const void *left, const void *right) {
    my_type l = *(my_type *) left, r = *(my_type *) right;
    return l < r ? -1 : l == r ? 0 : 1;
  }
.ve
  Then pass the function
.vb
  PetscTimSort(n, arr, sizeof(arr[0]), my_increasing_comparison_function)
.ve

  Notes:
  The comparison function must follow the qsort() comparison function paradigm, returning the signed difference between
  its arguments. If left < right : return -1, if left == right : return 0, if left > right : return 1. The user may also
 change or reverse the order of the sort by flipping the above. Note that stability of the sort is only guaranteed if
 the comparison function forms a valid trigraph.

  Timsort makes the assumption that input data is already likely partially ordered, or that it contains contiguous
  sections (termed 'runs') where the data is locally ordered (but not necessarily globally ordered). It therefore aims
 to select slices of the array in such a way that resulting mergesorts operate on near perfectly length-balanced
 arrays. To do so it repeatedly triggers attempts throughout to merge adjacent runs.

  Should one run continuously "win" a comparison the algorithm begins the "gallop" phase. It will aggressively
  search the "winner" for the location of the "losers" next entry (and vice versa) to copy all preceding elements into
  place in bulk. However if the data is truly unordered (as is the case with random data) the immense gains possible
  from these searches are expected __not__ to repay their costs. While adjacent arrays are almost all nearly the same
  size, they likely all contain similar data.

  A detailed description of the algorithm may be found here: https://bugs.python.org/file4451/timsort.txt

*/
PetscErrorCode PetscTimSort(PetscInt n, void *arr, size_t size, int (*cmp)(const void *, const void *))
{
  PetscInt           stacksize = 0, minrun, runstart = 0, runend = 0;
  PetscTimSortStack  runstack[128];
  PetscTimSortBuffer buff;
  PetscErrorCode     ierr;
  /* stacksize  = log_phi(n) = log_2(n)/log_2(phi), so 128 is enough for ~5.614e26 elements.
   It is so unlikely that this limit is reached that this is __never__ checked for */

  PetscFunctionBegin;
  /* Compute minrun. Minrun should be (32, 65) such that N/minrun
   is a power of 2 or one plus a power of 2 */
  {
    PetscInt t = n, r = 0;
    /* r becomes 1 if the least significant bits contain at least one off bit */
    while (t >= 64) {
      r |= t & 1;
      t >>= 1;
    }
    minrun = t + r;
  }
  if (PetscUnlikelyDebug(minrun < 32 || minrun > 65)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Calculated minrun %D not in range (32,65)",minrun);
  ierr = PetscInfo1(NULL, "minrun = %D\n", minrun);CHKERRQ(ierr);
  ierr = PetscMalloc1(minrun*size, &buff.ptr);CHKERRQ(ierr);
  buff.size = minrun*size;
  buff.maxsize = n*size;
  MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;
  while (runstart < n) {
    /* Check if additional entries are at least partially ordered and build natural run */
    ierr = PetscTimSortBuildRun_Private(n, arr, buff.ptr, size, cmp, minrun, runstart, &runend);CHKERRQ(ierr);
    runstack[stacksize].start = runstart;
    runstack[stacksize].size = runend-runstart+1;
    ierr = PetscTimSortMergeCollapse_Private(arr, size, cmp, &buff, runstack, &stacksize);CHKERRQ(ierr);
    ++stacksize;
    runstart = runend+1;
  }
  /* Have been inside while, so discard last stacksize++ */
  --stacksize;
  ierr = PetscTimSortForceCollapse_Private(arr, size, cmp, &buff, runstack, stacksize);CHKERRQ(ierr);
  ierr = PetscFree(buff.ptr);CHKERRQ(ierr);
  MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscIntSortSemiOrdered(PetscInt n, PetscInt arr[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTimSort(n, arr, sizeof(arr[0]), Compare_PetscInt_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
