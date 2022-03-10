#ifndef LEGACYCUBLASAPI_H
#define LEGACYCUBLASAPI_H

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnCpotrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnCpotrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnCpotrs((a),(b),(c),(d),(cuComplex*)(e),(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnCpotri((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnCpotri_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnCsytrf((a),(b),(c),(cuComplex*)(d),(e),(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnCsytrf_bufferSize((a),(b),(cuComplex*)(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnCgetrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnCgetrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnCgetrs((a),(b),(c),(d),(cuComplex*)(e),(f),(g),(cuComplex*)(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnCgeqrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnCgeqrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnCunmqr_bufferSize((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(h),(cuComplex*)(i),(cuComplex*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnCunmqr((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(h),(cuComplex*)(i),(cuComplex*)(j),(k),(cuComplex*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasCtrsm((a),(b),(c),(d),(e),(f),(g),(cuComplex*)(h),(cuComplex*)(i),(j),(cuComplex*)(k),(l))
#else /* complex double */
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnZpotrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnZpotrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnZpotrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnZpotri((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnZpotri_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnZsytrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnZsytrf_bufferSize((a),(b),(cuDoubleComplex*)(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnZgetrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnZgetrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnZgetrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(cuDoubleComplex*)(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnZgeqrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnZgeqrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnZunmqr_bufferSize((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(cuDoubleComplex*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnZunmqr((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(cuDoubleComplex*)(j),(k),(cuDoubleComplex*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasZtrsm((a),(b),(c),(d),(e),(f),(g),(cuDoubleComplex*)(h),(cuDoubleComplex*)(i),(j),(cuDoubleComplex*)(k),(l))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnSpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnSpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnSpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnSpotri((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnSpotri_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnSsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnSsytrf_bufferSize((a),(b),(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnSgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnSgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnSgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnSgeqrf_bufferSize((a),(b),(c),(float*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnSgeqrf((a),(b),(c),(float*)(d),(e),(float*)(f),(float*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnSormqr_bufferSize((a),(b),(c),(d),(e),(f),(float*)(g),(h),(float*)(i),(float*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnSormqr((a),(b),(c),(d),(e),(f),(float*)(g),(h),(float*)(i),(float*)(j),(k),(float*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasStrsm((a),(b),(c),(d),(e),(f),(g),(float*)(h),(float*)(i),(j),(float*)(k),(l))
#else /* real double */
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnDpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnDpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnDpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnDpotri((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnDpotri_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnDsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnDsytrf_bufferSize((a),(b),(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnDgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnDgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnDgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnDgeqrf_bufferSize((a),(b),(c),(double*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnDgeqrf((a),(b),(c),(double*)(d),(e),(double*)(f),(double*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnDormqr_bufferSize((a),(b),(c),(d),(e),(f),(double*)(g),(h),(double*)(i),(double*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnDormqr((a),(b),(c),(d),(e),(f),(double*)(g),(h),(double*)(i),(double*)(j),(k),(double*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasDtrsm((a),(b),(c),(d),(e),(f),(g),(double*)(h),(double*)(i),(j),(double*)(k),(l))
#endif
#endif

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXaxpy(a,b,c,d,e,f,g)               cublasCaxpy((a),(b),(cuComplex*)(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g))
#define cublasXscal(a,b,c,d,e)                   cublasCscal((a),(b),(cuComplex*)(c),(cuComplex*)(d),(e))
#define cublasXdotu(a,b,c,d,e,f,g)               cublasCdotu((a),(b),(cuComplex*)(c),(d),(cuComplex*)(e),(f),(cuComplex*)(g))
#define cublasXdot(a,b,c,d,e,f,g)                cublasCdotc((a),(b),(cuComplex*)(c),(d),(cuComplex*)(e),(f),(cuComplex*)(g))
#define cublasXswap(a,b,c,d,e,f)                 cublasCswap((a),(b),(cuComplex*)(c),(d),(cuComplex*)(e),(f))
#define cublasXnrm2(a,b,c,d,e)                   cublasScnrm2((a),(b),(cuComplex*)(c),(d),(e))
#define cublasIXamax(a,b,c,d,e)                  cublasIcamax((a),(b),(cuComplex*)(c),(d),(e))
#define cublasXasum(a,b,c,d,e)                   cublasScasum((a),(b),(cuComplex*)(c),(d),(e))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l)     cublasCgemv((a),(b),(c),(d),(cuComplex*)(e),(cuComplex*)(f),(g),(cuComplex*)(h),(i),(cuComplex*)(j),(cuComplex*)(k),(l))
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasCgemm((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(cuComplex*)(h),(i),(cuComplex*)(j),(k),(cuComplex*)(l),(cuComplex*)(m),(n))
#define cublasXgeam(a,b,c,d,e,f,g,h,i,j,k,l,m)   cublasCgeam((a),(b),(c),(d),(e),(cuComplex*)(f),(cuComplex*)(g),(h),(cuComplex*)(i),(cuComplex*)(j),(k),(cuComplex*)(l),(m))
#else /* complex double */
#define cublasXaxpy(a,b,c,d,e,f,g)               cublasZaxpy((a),(b),(cuDoubleComplex*)(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g))
#define cublasXscal(a,b,c,d,e)                   cublasZscal((a),(b),(cuDoubleComplex*)(c),(cuDoubleComplex*)(d),(e))
#define cublasXdotu(a,b,c,d,e,f,g)               cublasZdotu((a),(b),(cuDoubleComplex*)(c),(d),(cuDoubleComplex*)(e),(f),(cuDoubleComplex*)(g))
#define cublasXdot(a,b,c,d,e,f,g)                cublasZdotc((a),(b),(cuDoubleComplex*)(c),(d),(cuDoubleComplex*)(e),(f),(cuDoubleComplex*)(g))
#define cublasXswap(a,b,c,d,e,f)                 cublasZswap((a),(b),(cuDoubleComplex*)(c),(d),(cuDoubleComplex*)(e),(f))
#define cublasXnrm2(a,b,c,d,e)                   cublasDznrm2((a),(b),(cuDoubleComplex*)(c),(d),(e))
#define cublasIXamax(a,b,c,d,e)                  cublasIzamax((a),(b),(cuDoubleComplex*)(c),(d),(e))
#define cublasXasum(a,b,c,d,e)                   cublasDzasum((a),(b),(cuDoubleComplex*)(c),(d),(e))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l)     cublasZgemv((a),(b),(c),(d),(cuDoubleComplex*)(e),(cuDoubleComplex*)(f),(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(cuDoubleComplex*)(k),(l))
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasZgemm((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(k),(cuDoubleComplex*)(l),(cuDoubleComplex*)(m),(n))
#define cublasXgeam(a,b,c,d,e,f,g,h,i,j,k,l,m)   cublasZgeam((a),(b),(c),(d),(e),(cuDoubleComplex*)(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(cuDoubleComplex*)(j),(k),(cuDoubleComplex*)(l),(m))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXaxpy  cublasSaxpy
#define cublasXscal  cublasSscal
#define cublasXdotu  cublasSdot
#define cublasXdot   cublasSdot
#define cublasXswap  cublasSswap
#define cublasXnrm2  cublasSnrm2
#define cublasIXamax cublasIsamax
#define cublasXasum  cublasSasum
#define cublasXgemv  cublasSgemv
#define cublasXgemm  cublasSgemm
#define cublasXgeam  cublasSgeam
#else /* real double */
#define cublasXaxpy  cublasDaxpy
#define cublasXscal  cublasDscal
#define cublasXdotu  cublasDdot
#define cublasXdot   cublasDdot
#define cublasXswap  cublasDswap
#define cublasXnrm2  cublasDnrm2
#define cublasIXamax cublasIdamax
#define cublasXasum  cublasDasum
#define cublasXgemv  cublasDgemv
#define cublasXgemm  cublasDgemm
#define cublasXgeam  cublasDgeam
#endif
#endif

#endif // LEGACYCUBLASAPI_H
