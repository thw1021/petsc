#ifndef LEGACYCUBLASAPI_H
#define LEGACYCUBLASAPI_H

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
