#pragma once

/*
    The pragma below silence all compiler warnings coming from code in this header file.
    In particular, it silences `-Wfloat-equal` warnings in `operator==()` and `operator!=` below.
    Other compilers beyond GCC support this pragma.
*/
#if defined(__GNUC__) && (__GNUC__ >= 4) && !defined(__NEC__)
  #pragma GCC system_header
#endif

/*
     Defines additional operator overloading for the C++ complex class that are "missing" in the standard
     include files. For example, the code fragment

     std::complex<double> c = 22.0;
     c = 11 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'int' and 'std::complex<double>')

     The code fragment

     std::complex<float> c = 22.0;
     c = 11.0 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'double' and 'std::complex<float>')

     This deficiency means one may need to write cumbersome code while working with the C++ complex classes.

     This include file defines a few additional operator overload methods for the C++ complex classes to handle
     these cases naturally within PETSc code.

     This file is included in petscsystypes.h when feasible. In the small number of cases where these additional methods
     may conflict with other code one may add '#define PETSC_SKIP_CXX_COMPLEX_FIX 1' before including any PETSc include
     files to prevent these methods from being provided.
*/

#define PETSC_CXX_COMPLEX_FIX(Type) \
  static inline PetscComplex operator+(const PetscComplex &lhs, const Type &rhs) \
  { \
    return const_cast<PetscComplex &>(lhs) + PetscReal(rhs); \
  } \
  static inline PetscComplex operator+(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) + const_cast<PetscComplex &>(rhs); \
  } \
  static inline PetscComplex operator-(const PetscComplex &lhs, const Type &rhs) \
  { \
    return const_cast<PetscComplex &>(lhs) - PetscReal(rhs); \
  } \
  static inline PetscComplex operator-(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) - const_cast<PetscComplex &>(rhs); \
  } \
  static inline PetscComplex operator*(const PetscComplex &lhs, const Type &rhs) \
  { \
    return const_cast<PetscComplex &>(lhs) * PetscReal(rhs); \
  } \
  static inline PetscComplex operator*(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) * const_cast<PetscComplex &>(rhs); \
  } \
  static inline PetscComplex operator/(const PetscComplex &lhs, const Type &rhs) \
  { \
    return const_cast<PetscComplex &>(lhs) / PetscReal(rhs); \
  } \
  static inline PetscComplex operator/(const Type &lhs, const PetscComplex &rhs) \
  { \
    return PetscReal(lhs) / const_cast<PetscComplex &>(rhs); \
  } \
  static inline bool operator==(const PetscComplex &lhs, const Type &rhs) \
  { \
    return const_cast<PetscComplex &>(lhs).imag() == PetscReal(0) && const_cast<PetscComplex &>(lhs).real() == PetscReal(rhs); \
  } \
  static inline bool operator==(const Type &lhs, const PetscComplex &rhs) \
  { \
    return const_cast<PetscComplex &>(rhs).imag() == PetscReal(0) && const_cast<PetscComplex &>(rhs).real() == PetscReal(lhs); \
  } \
  static inline bool operator!=(const PetscComplex &lhs, const Type &rhs) \
  { \
    return const_cast<PetscComplex &>(lhs).imag() != PetscReal(0) || const_cast<PetscComplex &>(lhs).real() != PetscReal(rhs); \
  } \
  static inline bool operator!=(const Type &lhs, const PetscComplex &rhs) \
  { \
    return const_cast<PetscComplex &>(rhs).imag() != PetscReal(0) || const_cast<PetscComplex &>(rhs).real() != PetscReal(lhs); \
  } \
/* PETSC_CXX_COMPLEX_FIX */

// Provide PetscReal only, let C++ promote integers, floating points to PetscReals
PETSC_CXX_COMPLEX_FIX(PetscReal)
