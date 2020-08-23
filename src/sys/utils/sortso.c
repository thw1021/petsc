#include <petsc/private/petscimpl.h>
#include <petscsys.h>                /*I  "petscsys.h"  I*/

#define MIN_GALLOP_CONST_GLOBAL 8
static PetscInt MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;

#define _memcpy__(_a,_b,_size)__builtin_memcpy((_a),(_b),(_size));
#define _memmove__(_a,_b,_size) __builtin_memmove((_a),(_b),(_size));

typedef struct {
  PetscInt size;
  PetscInt start;
} PetscTimSortStack;

typedef struct {
  void     *ptr;
  size_t   size;
  size_t   maxsize;
} PetscTimSortBuffer;

PETSC_STATIC_INLINE PetscErrorCode PetscIntGallopSearchLeft_Private(const PetscInt[],PetscInt,PetscInt,PetscInt,PetscInt*);
PETSC_STATIC_INLINE PetscErrorCode PetscIntGallopSearchRight_Private(const PetscInt[],PetscInt,PetscInt,PetscInt,PetscInt*);
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortMergeLo_Private(PetscInt[],PetscInt*,PetscInt,PetscInt,PetscInt);
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortMergeHi_Private(PetscInt[],PetscInt*,PetscInt,PetscInt,PetscInt);
PETSC_STATIC_INLINE PetscErrorCode PetscIntInsertionSort_Private(PetscInt[],PetscInt,PetscInt,PetscInt);
PETSC_STATIC_INLINE PetscErrorCode PetscIntBinaryInsertionSort_Private(PetscInt[],PetscInt,PetscInt,PetscInt);
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortForceCollapse_Private(PetscInt[],PetscTimSortBuffer*,PetscTimSortStack*,PetscInt);
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortMergeCollapse_Private(PetscInt[],PetscTimSortBuffer*,PetscTimSortStack*,PetscInt*);
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortBuildRun_Private(PetscInt[],PetscInt,PetscInt,PetscInt,PetscInt*);

PETSC_STATIC_INLINE void _memcpy(void *dest, const void *src, size_t size)
{
  PetscFunctionBegin;
  if ((size) < 0) PetscPrintf(PETSC_COMM_SELF,"SIZE CPY %D <= 0\n",size);
  __builtin_memcpy((char *)dest, (char *)src, size);
  PetscFunctionReturnVoid();
}

PETSC_STATIC_INLINE void _memmove(void *dest, const void *src, size_t size)
{
  PetscFunctionBegin;
  if ((size) < 0) PetscPrintf(PETSC_COMM_SELF,"SIZE MOVE %D <= 0\n",size);
  __builtin_memmove((char *)dest, (char *)src, size);
  PetscFunctionReturnVoid();
}

#define PetscGallopSearchLeft_Agnostic(arr,l,r,x,m)                     \
  do {                                                                  \
    PetscInt last = l, k = 1, mid, cur = l+1;                           \
    if (PetscUnlikelyDebug(r < l)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"r %D < l %D in PetscGallopSearchLeft",r,l); \
    if (PetscUnlikely(!r-l)) {*m = l;PetscFunctionReturn(0);}           \
    if (x < arr[l]) {*m = l;PetscFunctionReturn(0);}                    \
    while (PETSC_TRUE) {                                                \
      if (cur > r) {cur = r; break;}                                    \
      if (x < arr[cur]) break;                                          \
      last = cur;                                                       \
      cur += (k <<= 1) + 1; ++k;                                        \
    }                                                                   \
    /* standard binary search but take last 0 mid 0 cur 1 into account*/ \
    while (cur > last + 1) {                                            \
      mid = last + ((cur - last) >> 1);                                 \
      if (x < arr[mid]) {                                               \
        cur = mid;                                                      \
      } else {                                                          \
        last = mid;                                                     \
      }                                                                 \
    }                                                                   \
    *m = cur;                                                           \
  } while (0)

#define PetscGallopSearchRight_Agnostic(arr,l,r,x,m)                    \
  do {                                                                  \
    PetscInt last = r, k = 1, mid, cur = r-1;                           \
    if (PetscUnlikelyDebug(r < l)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"r %D < l %D in PetscGallopSearchRight",r,l); \
    if (PetscUnlikely(!r-l)) {*m = r;PetscFunctionReturn(0);}           \
    if (x > arr[r]) {*m = r;PetscFunctionReturn(0);}                    \
    while (PETSC_TRUE) {                                                \
      if (cur < l) {cur = l; break;}                                    \
      if (x > arr[cur]) break;                                          \
      last = cur;                                                       \
      cur -= (k <<= 1) + 1; ++k;                                        \
    }                                                                   \
    /* standard binary search but take last r-1 mid r-1 cur r-2 into account*/ \
    while (last > cur + 1) {                                            \
      mid = last - ((last - cur) >> 1);                                 \
      if (x > arr[mid]) {                                               \
        cur = mid;                                                      \
      } else {                                                          \
        last = mid;                                                     \
      }                                                                 \
    }                                                                   \
    *m = cur;                                                           \
  } while (0)

#define PetscTimSortMergeLo_Agnostic(arr,tarr,left,mid,right,type)      \
  do {                                                                  \
    const PetscInt llen = mid-left;                                     \
    PetscInt       i = 0, j = mid, k = left, gallopleft = 0, gallopright = 0; \
    size_t         size = sizeof(type);                                 \
    PetscErrorCode ierr;                                                \
    _memcpy(tarr, arr+left, llen*size);                                 \
    while ((i < llen) && (j <= right)) {                                \
      if (tarr[i] < arr[j]) {                                           \
        arr[k++] = tarr[i++];                                           \
        gallopright = 0;                                                \
        if (++gallopleft >= MIN_GALLOP_GLOBAL && i < llen) {            \
          PetscInt l1, l2, diff1, diff2;                                \
          do {                                                          \
            if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;             \
            /* search temp for right[j], can move up to that of temp into arr immediately */ \
            ierr = type##GallopSearchLeft_Private(tarr, i, llen-1, arr[j], &l1);CHKERRQ(ierr); \
            diff1 = l1-i;                                               \
            _memcpy(arr+k, tarr+i, diff1*size);                         \
            k += diff1;                                                 \
            i = l1;                                                     \
            /* search right for temp[i], can move up to that many of right into arr */ \
            ierr = type##GallopSearchLeft_Private(arr, j, right, tarr[i], &l2);CHKERRQ(ierr); \
            diff2 = l2-j;                                               \
            _memmove(arr+k, arr+j, diff2*size);                         \
            k += diff2;                                                 \
            j = l2;                                                     \
          } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL); \
          ++MIN_GALLOP_GLOBAL;                                          \
        }                                                               \
      } else {                                                          \
        arr[k++] = arr[j++];                                            \
        gallopleft = 0;                                                 \
        if (++gallopright >= MIN_GALLOP_GLOBAL && j <= right) {         \
          PetscInt l1, l2, diff1, diff2;                                \
          do {                                                          \
            if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;             \
            /* search right for temp[i], can move up to that many of right into arr */ \
            ierr = type##GallopSearchLeft_Private(arr, j, right, tarr[i], &l2);CHKERRQ(ierr); \
            diff2 = l2-j;                                               \
            _memmove(arr+k, arr+j, diff2*size);                         \
            k += diff2;                                                 \
            j = l2;                                                     \
            /* search temp for right[j], can copy up to that of temp into arr immediately */ \
            ierr = type##GallopSearchLeft_Private(tarr, i, llen-1, arr[j], &l1);CHKERRQ(ierr); \
            diff1 = l1-i;                                               \
            _memcpy(arr+k, tarr+i, diff1*size);                         \
            k += diff1;                                                 \
            i = l1;                                                     \
          } while (diff1 > MIN_GALLOP_GLOBAL || diff1 > MIN_GALLOP_GLOBAL); \
          ++MIN_GALLOP_GLOBAL;                                          \
        }                                                               \
      }                                                                 \
      if (i<llen) _memcpy(arr+k, tarr+i, (llen-i)*size);                \
    }                                                                   \
  } while (0)

#define PetscTimSortMergeHi_Agnostic(arr,tarr,left,mid,right,type)      \
  do {                                                                  \
    const PetscInt rlen = right-mid+1;                                  \
    PetscInt       i = right-mid, j = mid-1, k = right, gallopleft = 0, gallopright = 0; \
    size_t         size = sizeof(type);                                 \
    PetscErrorCode ierr;                                                \
    _memcpy(tarr, arr+mid, rlen*size);                                  \
    while ((i >= 0) && (j >= left)) {                                   \
      if (tarr[i] > arr[j]) {                                           \
        arr[k--] = tarr[i--];                                           \
        gallopleft = 0;                                                 \
        if (++gallopright >= MIN_GALLOP_GLOBAL && i >= 0) {             \
          PetscInt l1, l2, diff1, diff2;                                \
          do {                                                          \
            if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;             \
            /* search temp for left[j], can copy up to that many of temp into arr */ \
            ierr = type##GallopSearchRight_Private(tarr, 0, i, arr[j], &l1);CHKERRQ(ierr); \
            diff1 = i-l1;                                               \
            _memcpy(arr+k-diff1+1, tarr+l1+1, diff1*size);              \
            k -= diff1;                                                 \
            i = l1;                                                     \
            /* search left for temp[i], can move up to that many of left up arr */ \
            ierr = type##GallopSearchRight_Private(arr, left, j, tarr[i], &l2);CHKERRQ(ierr); \
            diff2 = j-l2;                                               \
            _memmove(arr+k-diff2+1, arr+l2+1, diff2*size);              \
            k -= diff2;                                                 \
            j = l2;                                                     \
          } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL); \
          ++MIN_GALLOP_GLOBAL;                                          \
        }                                                               \
      } else {                                                          \
        arr[k--] = arr[j--];                                            \
        gallopright = 0;                                                \
        if (++gallopleft >= MIN_GALLOP_GLOBAL && j >= left) {           \
          PetscInt l1, l2, diff1, diff2;                                \
          do {                                                          \
            if (MIN_GALLOP_GLOBAL > 1) --MIN_GALLOP_GLOBAL;             \
            /* search left for temp[i], can move up to that many of left up arr */ \
            ierr = type##GallopSearchRight_Private(arr, left, j, tarr[i], &l2);CHKERRQ(ierr); \
            diff2 = j-l2;                                               \
            _memmove(arr+k-diff2+1, arr+l2+1, diff2*size);              \
            k -= diff2;                                                 \
            j = l2;                                                     \
            /* search temp for left[j], can copy up to that many of temp into arr */ \
            ierr = type##GallopSearchRight_Private(tarr, 0, i, arr[j], &l1);CHKERRQ(ierr); \
            diff1 = i-l1;                                               \
            _memcpy(arr+k-diff1+1, tarr+l1+1, diff1*size);              \
            k -= diff1;                                                 \
            i = l1;                                                     \
          } while (diff1 > MIN_GALLOP_GLOBAL || diff2 > MIN_GALLOP_GLOBAL); \
          ++MIN_GALLOP_GLOBAL;                                          \
        }                                                               \
      }                                                                 \
    }                                                                   \
    if (i >= 0) _memcpy(arr+left, tarr, (i+1)*size);                    \
  } while (0)

#define PetscInsertionSort_Agnostic(arr,left,start,right,type)    \
  do {                                                            \
    PetscInt i;                                                   \
    if (start == left) ++start;                                   \
    for (i = start; i <= right; ++i) {                            \
      const type t = arr[i];                                      \
      PetscInt   j = i-1;                                         \
      while ((j >= left) && (t < arr[j])) {                       \
        arr[j+1] = arr[j];                                        \
        --j;                                                      \
      }                                                           \
      arr[j+1] = t;                                               \
    }                                                             \
  } while (0)

#define PetscBinaryInsertionSort_Agnostic(arr,left,start,right,type)    \
  do {                                                                  \
    PetscInt i;                                                         \
    if (start == left) ++start;                                         \
    for (i = start; i <= right; ++i) {                                  \
      PetscInt   l = left, r = i;                                       \
      const type t = arr[i];                                            \
      do {                                                              \
        PetscInt m;                                                     \
        m = l + ((r - l) >> 1);                                         \
        if (t < arr[m]) {                                               \
          r = m;                                                        \
        } else {                                                        \
          l = m + 1;                                                    \
        }                                                               \
      } while (l < r);                                                  \
      _memmove(arr+l+1, arr+l, (i-l)*sizeof(type));                     \
      arr[l] = t;                                                       \
    }                                                                   \
  } while (0)

#define PetscTimSortForceCollapse_Agnostic(arr,buff,stack,stacksize,type) \
  while (stacksize) {                                                   \
    PetscInt       l, m = stack[stacksize].start, r;                    \
    PetscErrorCode ierr;                                                \
    /* A = stack[i-1], B = stack[i] */                                  \
    /* Search A for B[0] insertion */                                   \
    ierr = type##GallopSearchLeft_Private(arr, stack[stacksize-1].start, stack[stacksize].start-1, arr[stack[stacksize].start], &l);CHKERRQ(ierr); \
    /* l == m-1 means sorted */                                         \
    if (l < m-1) {                                                      \
      /* Search B for A[-1] insertion */                                \
      ierr = type##GallopSearchRight_Private(arr, stack[stacksize].start, stack[stacksize].start+stack[stacksize].size-1, arr[stack[stacksize].start-1], &r);CHKERRQ(ierr); \
      if (m-l <= r-m) {                                                 \
        ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*sizeof(type));CHKERRQ(ierr); \
        ierr = type##TimSortMergeLo_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
      } else {                                                          \
        ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*sizeof(type));CHKERRQ(ierr); \
        ierr = type##TimSortMergeHi_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
      }                                                                 \
    }                                                                   \
    /* Update A with merge */                                           \
    stack[stacksize-1].size += stack[stacksize].size;                   \
    --stacksize;                                                        \
  }

#define PetscTimSortMergeCollapse_Agnostic(arr,buff,stack,stacksize,type) \
  do {                                                                  \
    PetscInt i = *stacksize;                                            \
    while (i) {                                                         \
      PetscErrorCode ierr;                                              \
      PetscInt       l, m, r, itemp = i;                                \
      if (i == 1) {                                                     \
        /* A = stack[i-1], B = stack[i] */                              \
        if (stack[i-1].size < stack[i].size) {                          \
          m = stack[i].start;                                           \
          /* Search A for B[0] insertion */                             \
          ierr = type##GallopSearchLeft_Private(arr, stack[i-1].start, stack[i].start-1, arr[stack[i].start], &l);CHKERRQ(ierr); \
          /* l == m-1 means sorted */                                   \
          if (l < m-1) {                                                \
            /* Search B for A[-1] insertion */                          \
            ierr = type##GallopSearchRight_Private(arr, stack[i].start, stack[i].start+stack[i].size-1, arr[stack[i].start-1], &r);CHKERRQ(ierr); \
            if (m-l <= r-m) {                                           \
              ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*sizeof(type));CHKERRQ(ierr); \
              ierr = type##TimSortMergeLo_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
            } else {                                                    \
              ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*sizeof(type));CHKERRQ(ierr); \
              ierr = type##TimSortMergeHi_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
            }                                                           \
          }                                                             \
          /* Update A with merge */                                     \
          stack[i-1].size += stack[i].size;                             \
          --i;                                                          \
        }                                                               \
      } else {                                                          \
        /* i > 2, i.e. C exists */                                      \
        /* A = stack[i-2], B = stack[i-1], C = stack[i]; */             \
        if (stack[i-2].size <= stack[i-1].size+stack[i].size) {         \
          if (stack[i-2].size < stack[i].size) {                        \
            /* merge B into A */                                        \
            m = stack[i-1].start;                                       \
            /* Search A for B[0] insertion */                           \
            ierr = type##GallopSearchLeft_Private(arr, stack[i-2].start, stack[i-1].start-1, arr[stack[i-1].start], &l);CHKERRQ(ierr); \
            if (l < m-1) {                                              \
              /* Search B for A[-1] insertion */                        \
              ierr = type##GallopSearchRight_Private(arr, stack[i-1].start, stack[i-1].start+stack[i-1].size-1, arr[stack[i-1].start-1], &r);CHKERRQ(ierr); \
              if (m-l <= r-m) {                                         \
                ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*sizeof(type));CHKERRQ(ierr); \
                ierr = type##TimSortMergeLo_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
              } else {                                                  \
                ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*sizeof(type));CHKERRQ(ierr); \
                ierr = type##TimSortMergeHi_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
              }                                                         \
            }                                                           \
            /* Update A with merge */                                   \
            stack[i-2].size += stack[i-1].size;                         \
            /* Push C up the stack */                                   \
            stack[i-1].start = stack[i].start;                          \
            stack[i-1].size = stack[i].size;                            \
          } else {                                                      \
            /* merge C into B */                                        \
            mergeBC:                                                    \
            m = stack[i].start;                                         \
            /* Search B for C[0] insertion */                           \
            ierr = type##GallopSearchLeft_Private(arr, stack[i-1].start, stack[i].start-1, arr[stack[i].start], &l);CHKERRQ(ierr); \
            if (l < m-1) {                                              \
              /* Search C for B[-1] insertion */                        \
              ierr = type##GallopSearchRight_Private(arr, stack[i].start, stack[i].start+stack[i].size-1, arr[stack[i].start-1], &r);CHKERRQ(ierr); \
              if (m-l <= r-m) {                                         \
                ierr = PetscTimSortResizeBuffer_Private(buff, (m-l+1)*sizeof(type));CHKERRQ(ierr); \
                ierr = type##TimSortMergeLo_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
              } else {                                                  \
                ierr = PetscTimSortResizeBuffer_Private(buff, (r-m+1)*sizeof(type));CHKERRQ(ierr); \
                ierr = type##TimSortMergeHi_Private(arr, buff->ptr, l, m, r);CHKERRQ(ierr); \
              }                                                         \
            }                                                           \
            /* Update B with merge */                                   \
            stack[i-1].size += stack[i].size;                           \
          }                                                             \
          --i;                                                          \
        } else if (stack[i-1].size <= stack[i].size) {                  \
          /* merge C into B */                                          \
          goto mergeBC;                                                 \
        }                                                               \
      }                                                                 \
      if (itemp == i) break;                                            \
    }                                                                   \
    *stacksize = i;                                                     \
  } while (0)

#define PetscTimSortBuildRun_Agnostic(n,arr,minrun,runstart,runend,type) \
  do {                                                                  \
    const PetscInt  re = PetscMin(runstart+minrun, n-1);                \
    PetscInt        ri = runstart;                                      \
    if (PetscUnlikely(runstart == n-1)) {*runend = runstart; PetscFunctionReturn(0);} \
    /* guess whether run is ascending or descending and tally up the */ \
    /* longest consecutive run. essentially a coinflip for random data */ \
    if (arr[ri+1] < arr[ri]) {                                          \
      ++ri;                                                             \
      while (ri < n-1) {                                                \
        if (arr[ri+1] >= arr[ri]) break;                                \
        ++ri;                                                           \
      }                                                                 \
      {                                                                 \
        PetscInt lo = runstart, hi = ri;                                \
        type     t;                                                     \
        do {                                                            \
          t = arr[hi]; arr[hi] = arr[lo]; arr[lo] = t;                  \
          ++lo; --hi;                                                   \
        } while (lo < hi);                                              \
      }                                                                 \
    } else {                                                            \
      ++ri;                                                             \
      while (ri < n-1) {                                                \
        if (arr[ri+1] < arr[ri]) break;                                 \
        ++ri;                                                           \
      }                                                                 \
    }                                                                   \
    if (PetscDefined(USE_DEBUG)) {                                      \
      PetscErrorCode ierr;                                              \
      ierr = PetscInfo1(NULL, "natural run length = %D\n", ri-runstart+1);CHKERRQ(ierr); \
    }                                                                   \
    if (ri < re) {                                                      \
      /* the attempt failed, this section likely contains random data */ \
      /* If ri got close to minrun (within 50%) then we try binary search */ \
      if (ri-runstart <= minrun >> 1) {                                 \
        ++MIN_GALLOP_GLOBAL; /* didn't get close hedge our bets against random data */ \
        type##InsertionSort_Private(arr, runstart, ri, re);             \
      } else {                                                          \
        type##BinaryInsertionSort_Private(arr, runstart, ri, re);       \
      }                                                                 \
      *runend = re;                                                     \
    } else *runend = ri;                                                \
  } while (0)

#define PetscTimSort_Agnostic(n,arr,type)                               \
  do {                                                                  \
    PetscInt           stacksize = 0, minrun, runstart = 0, runend = 0; \
    PetscTimSortStack  runstack[128];                                   \
    /* stack grows like log_phi(n) so this is enough for roughly 5.6e26 */ \
    /* this is so incredibly large its never checked for*/              \
    PetscTimSortBuffer buff;                                            \
    PetscErrorCode     ierr;                                            \
    /* Compute minrun. Minrun should be (32, 65) such that N/minrun */  \
    /* is a power of 2 or one plus a power of 2 */                      \
    {                                                                   \
      PetscInt t = (n), r = 0;                                          \
      /* r becomes 1 if the least significant bits contain at least one off bit */ \
      while (t >= 64) {                                                 \
        r |= t & 1;                                                     \
        t >>= 1;                                                        \
      }                                                                 \
      minrun = t + r;                                                   \
    }                                                                   \
    if (PetscUnlikelyDebug(minrun < 32 || minrun > 65)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Calculated minrun %D not in range (32,65)",minrun); \
    ierr = PetscInfo1(NULL, "minrun = %D\n", minrun);CHKERRQ(ierr);     \
    ierr = PetscMalloc1(minrun*sizeof(type), &buff.ptr);CHKERRQ(ierr);  \
    buff.size = minrun*sizeof(type);                                    \
    buff.maxsize = (n)*sizeof(type);                                    \
    MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;                        \
    while (runstart < (n)) {                                            \
      /* Check if additional entries are at least partially ordered and build natural run */ \
      ierr = type##TimSortBuildRun_Private((arr), (n), minrun, runstart, &runend);CHKERRQ(ierr); \
      runstack[stacksize].start = runstart;                             \
      runstack[stacksize].size = runend-runstart+1;                     \
      ierr = type##TimSortMergeCollapse_Private((arr), &buff, runstack, &stacksize);CHKERRQ(ierr); \
      ++stacksize;                                                      \
      runstart = runend+1;                                              \
    }                                                                   \
    /* Have been inside while, so discard last stacksize++ */           \
    --stacksize;                                                        \
    ierr = type##TimSortForceCollapse_Private((arr), &buff, runstack, stacksize);CHKERRQ(ierr); \
    ierr = PetscFree(buff.ptr);CHKERRQ(ierr);                           \
    MIN_GALLOP_GLOBAL = MIN_GALLOP_CONST_GLOBAL;                        \
  } while (0)

PETSC_STATIC_INLINE PetscErrorCode PetscTimSortResizeBuffer_Private(PetscTimSortBuffer *buff, PetscInt newSize)
{
  PetscFunctionBegin;
  if (PetscLikely(newSize <= buff->size)) PetscFunctionReturn(0);
  {
    /* Can't be larger than n, there is merit to simply allocating buff to n to begin with */
    PetscErrorCode ierr;
    PetscInt       newMax = PetscMin(newSize*newSize, buff->maxsize);
    ierr = PetscFree(buff->ptr);CHKERRQ(ierr);
    ierr = PetscMalloc1(newMax, &buff->ptr);CHKERRQ(ierr);
    buff->size = newMax;
  }
  PetscFunctionReturn(0);
}

/* Start left look right. Looking for e.g. B[0] in A or mergelo. l inclusive, r inclusive. Returns first m such that arr[m] >
 x. Output also inclusive */
PETSC_STATIC_INLINE PetscErrorCode PetscIntGallopSearchLeft_Private(const PetscInt arr[], PetscInt l, PetscInt r, PetscInt x, PetscInt *m)
{
  PetscFunctionBegin;
  PetscGallopSearchLeft_Agnostic(arr,l,r,x,m);
  PetscFunctionReturn(0);
}

/* Start right look left. Looking for e.g. B[-1] in A or mergehi. l inclusive, r inclusive. Returns last m such that arr[m]
 < x. Output also inclusive */
PETSC_STATIC_INLINE PetscErrorCode PetscIntGallopSearchRight_Private(const PetscInt arr[], PetscInt l, PetscInt r, PetscInt x, PetscInt *m)
{
  PetscFunctionBegin;
  PetscGallopSearchRight_Agnostic(arr,l,r,x,m);
  PetscFunctionReturn(0);
}

/* Mergesort where size of left half <= size of right half, so mergesort is done left to right. Arr should be pointer to
 complete array, left is first index of left array, mid is first index of right array, right is last index of right
 array */
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortMergeLo_Private(PetscInt arr[], PetscInt *tarr, PetscInt left, PetscInt mid, PetscInt right)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  PetscErrorCode ierr;
  PetscInt *temp;
  const PetscInt n = right-left+1;
  ierr = PetscMalloc1(n, &temp);CHKERRQ(ierr);
  ierr = PetscArraycpy(temp, arr+left, n);CHKERRQ(ierr);
  ierr = PetscSortInt(n, temp);CHKERRQ(ierr);
#endif
  PetscTimSortMergeLo_Agnostic(arr,tarr,left,mid,right,PetscInt);
#if defined(PETSC_USE_DEBUG)
  for (PetscInt i = 0; i < n; ++i) {
    if (temp[i] != arr[left+i]) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"LO temp[%D] = %D != arr[%D] = %D!",i,temp[i],i+left,arr[left+i]);
    }
  }
  ierr = PetscFree(temp);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* Mergesort where size of right half < size of left half, so mergesort is done right to left. Arr should be pointer to
 complete array, left is first index of left array, mid is first index of right array, right is last index of right
 array */
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortMergeHi_Private(PetscInt arr[], PetscInt *tarr, PetscInt left, PetscInt mid, PetscInt right)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  PetscErrorCode ierr;
  PetscInt *temp;
  const PetscInt n = right-left+1;
  ierr = PetscMalloc1(n, &temp);CHKERRQ(ierr);
  ierr = PetscArraycpy(temp, arr+left, n);CHKERRQ(ierr);
  ierr = PetscSortInt(n, temp);CHKERRQ(ierr);
#endif
  PetscTimSortMergeHi_Agnostic(arr,tarr,left,mid,right,PetscInt);
#if defined(PETSC_USE_DEBUG)
  for (PetscInt i = 0; i < n; ++i) {
    if (temp[i] != arr[left+i]) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"HI temp[%D] = %D != arr[%D] = %D!",i,temp[i],i+left,arr[left+i]);
    }
  }
  ierr = PetscFree(temp);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* Left is inclusive lower bound of array slice, start is start location of unsorted section, right is inclusive upper
 bound of array slice. If unsure of where unsorted section starts or if entire length is unsorted pass start = left */
PETSC_STATIC_INLINE PetscErrorCode PetscIntInsertionSort_Private(PetscInt arr[], PetscInt left, PetscInt start, PetscInt right)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  PetscErrorCode ierr;
  PetscInt *temp;
  const PetscInt n = right-left+1;
  ierr = PetscMalloc1(n, &temp);CHKERRQ(ierr);
  ierr = PetscArraycpy(temp, arr+left, n);CHKERRQ(ierr);
  ierr = PetscSortInt(n, temp);CHKERRQ(ierr);
#endif
  PetscInsertionSort_Agnostic(arr,left,start,right,PetscInt);
#if defined(PETSC_USE_DEBUG)
  for (PetscInt i = 0; i < n; ++i) {
    if (temp[i] != arr[left+i]) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"INSORT temp[%D] = %D != arr[%D] = %D!",i,temp[i],i+left,arr[left+i]);
    }
  }
  ierr = PetscFree(temp);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* See PetscInsertionSort_Private */
PETSC_STATIC_INLINE PetscErrorCode PetscIntBinaryInsertionSort_Private(PetscInt arr[], PetscInt left, PetscInt start, PetscInt right)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  PetscErrorCode ierr;
  PetscInt *temp;
  const PetscInt n = right-left+1;
  ierr = PetscMalloc1(n, &temp);CHKERRQ(ierr);
  ierr = PetscArraycpy(temp, arr+left, n);CHKERRQ(ierr);
  ierr = PetscSortInt(n, temp);CHKERRQ(ierr);
#endif
  PetscBinaryInsertionSort_Agnostic(arr,left,start,right,PetscInt);
#if defined(PETSC_USE_DEBUG)
  for (PetscInt i = 0; i < n; ++i) {
    if (temp[i] != arr[left+i]) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"BIN INSORT temp[%D] = %D != arr[%D] = %D!",i,temp[i],i+left,arr[left+i]);
    }
  }
  ierr = PetscFree(temp);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortForceCollapse_Private(PetscInt arr[], PetscTimSortBuffer *buff, PetscTimSortStack *stack, PetscInt stacksize)
{
  PetscFunctionBegin;
  PetscTimSortForceCollapse_Agnostic(arr,buff,stack,stacksize,PetscInt);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortMergeCollapse_Private(PetscInt arr[], PetscTimSortBuffer *buff, PetscTimSortStack *stack, PetscInt *stacksize)
{
  PetscFunctionBegin;
  PetscTimSortMergeCollapse_Agnostic(arr,buff,stack,stacksize,PetscInt);
  PetscFunctionReturn(0);
}

/* March sequentially through the array building up a "run" of weakly increasing or strictly decreasing contiguous
 elements. Decreasing runs are reversed by swapping. If the run is less than minrun, artificially extend it via either
 binary insertion sort or regulat insertion sort */
PETSC_STATIC_INLINE PetscErrorCode PetscIntTimSortBuildRun_Private(PetscInt arr[], PetscInt n, PetscInt minrun, PetscInt runstart, PetscInt *runend)
{
  PetscFunctionBegin;
  PetscTimSortBuildRun_Agnostic(n,arr,minrun,runstart,runend,PetscInt);
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
PetscErrorCode PetscTimSort(PetscInt n, PetscInt arr[])
{
  PetscFunctionBegin;
  PetscValidIntPointer(arr,2);
  PetscTimSort_Agnostic(n,arr,PetscInt);
  PetscFunctionReturn(0);
}

/*@
 PetscIntSortSemiOrdered - Sorts an array of integers in place in increasing order.

 Not Collective

 Input Parameters:
 +  n   - number of values
 -  arr - array of integers

 Output Parameters:
 .  arr - sorted array of integers

 Notes:
 If the array is less than 64 entries long PetscSortMPIInt() is automatically used.

 This function serves as an alternative to PetscSortInt(). While this function works for any array of integers it is
 significantly faster if the array is not totally random. There are exceptions to this and so it is __highly__
 recomended that the user benchmark their code to see which routine is fastest.

 Level: intermediate

 .seealso: PetscTimSort(), PetscSortInt(), PetscSortIntWithPermutation()
 @*/
PetscErrorCode PetscIntSortSemiOrdered(PetscInt n, PetscInt arr[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(arr,2);
  if (n == 1) PetscFunctionReturn(0);
  if (n < 64) {
    ierr = PetscSortInt(n, arr);CHKERRQ(ierr);
  } else {
    PetscTimSort_Agnostic(n,arr,PetscInt);
  }
  PetscFunctionReturn(0);
}

/*@
 PetscMPIIntSortSemiOrdered - Sorts an array of PetscMPIInts in place in increasing order.

 Not Collective

 Input Parameters:
 +  n   - number of values
 -  arr - array of PetscMPIInts

 Output Parameters:
 .  arr - sorted array of integers

 Notes:
 If the array is less than 64 entries long PetscSortMPIInt() is automatically used.

 This function serves as an alternative to PetscSortMPIInt(). While this function works for any array of PetscMPIInts it is
 significantly faster if the array is not totally random. There are exceptions to this and so it is __highly__
 recomended that the user benchmark their code to see which routine is fastest.

 Level: intermediate

 .seealso: PetscTimSort(), PetscSortMPIInt()
 @*/
PetscErrorCode PetscMPIIntSortSemiOrdered(PetscInt n, PetscMPIInt arr[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(arr,2);
  if (n == 1) PetscFunctionReturn(0);
  if (n < 64) {
    ierr = PetscSortMPIInt(n, arr);CHKERRQ(ierr);
  } else {
    ierr = PetscTimSort(n, arr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
