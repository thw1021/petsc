#if !defined(PETSC4PY_PYSCALAR_H)
#define PETSC4PY_PYSCALAR_H

#include <Python.h>
#include <petscsystypes.h>

#if PetscDefined(HAVE_COMPLEX)

static inline
PyObject *PyPetscComplex_FromPetscComplex(PetscComplex s)
{
  double a = (double)PetscRealPartComplex(s);
  double b = (double)PetscImaginaryPartComplex(s);
  return PyComplex_FromDoubles(a, b);
}

static inline
PetscComplex PyPetscComplex_AsPetscComplex(PyObject *o)
{
#if defined(Py_LIMITED_API)
  PetscReal a = (PetscReal)PyComplex_RealAsDouble(o);
  PetscReal b = (PetscReal)PyComplex_ImagAsDouble(o);
#else
  Py_complex cval = PyComplex_AsCComplex(o);
  PetscReal a = (PetscReal)cval.real;
  PetscReal b = (PetscReal)cval.imag;
#endif
  return a + b * PETSC_i;
}

#else

#if !defined(PetscComplex)
typedef struct { PetscReal real, imag; } _py_PetscComplex;
#define PetscComplex _py_PetscComplex
#endif

static inline
PyObject *PyPetscComplex_FromPetscComplex(PetscComplex s)
{
  (void) s;
  PyErr_SetString(PyExc_TypeError, "PETSc built without complex numbers");
  return NULL;
}

static inline
PetscComplex PyPetscComplex_AsPetscComplex(PyObject *o)
{
  PetscComplex c = {-1, 0};
  PyErr_SetString(PyExc_TypeError, "PETSc built without complex numbers");
  return c;
}

#endif

static inline
PyObject *PyPetscScalar_FromPetscScalar(PetscScalar s)
{
#if PetscDefined(USE_COMPLEX)
  double a = (double)PetscRealPart(s);
  double b = (double)PetscImaginaryPart(s);
  return PyComplex_FromDoubles(a, b);
#else
  return PyFloat_FromDouble((double)s);
#endif
}

static inline
PetscScalar PyPetscScalar_AsPetscScalar(PyObject *o)
{
#if PetscDefined(USE_COMPLEX)
#if defined(Py_LIMITED_API)
  PetscReal a = (PetscReal)PyComplex_RealAsDouble(o);
  PetscReal b = (PetscReal)PyComplex_ImagAsDouble(o);
#else
  Py_complex cval = PyComplex_AsCComplex(o);
  PetscReal a = (PetscReal)cval.real;
  PetscReal b = (PetscReal)cval.imag;
#endif
  return a + b * PETSC_i;
#else
  return (PetscScalar)PyFloat_AsDouble(o);
#endif
}

#endif/*PETSC4PY_PYSCALAR_H*/
