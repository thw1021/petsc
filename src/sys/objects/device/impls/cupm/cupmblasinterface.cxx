#include <petsc/private/cupmblasinterface.hpp>

namespace Petsc
{

namespace Impl
{

#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT_0(PREFIX,ORIGINAL,MAPPED)
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT_1(PREFIX,ORIGINAL,MAPPED) \
  const decltype(ORIGINAL) CUPMBlasInterface<CUPMDeviceType::PREFIX>::MAPPED;

#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT(HAVE,PREFIX,ORGINAL,MAPPED) \
  PETSC_CONCAT(                                                         \
    PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT_,               \
    HAVE                                                                \
  )(PREFIX,ORGINAL,MAPPED)

#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT_(PREFIX,ORGINAL,MAPPED) \
  PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_IF_HAVE_EXACT(                  \
    PetscDefined(PETSC_CONCAT(HAVE_,PREFIX)),                           \
    PREFIX,ORGINAL,                                                     \
    MAPPED                                                              \
  )

// in case either one or the other don't agree on a name, you can specify all three here
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(CUORIGINAL,HIPORIGINAL,MAPPED) \
  PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT_(                         \
    CUDA,                                                               \
    PETSC_CONCAT(CUBLAS,CUORIGINAL),                                    \
    PETSC_CONCAT(CUPMBLAS,MAPPED)                                       \
  )                                                                     \
  PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT_(                         \
    HIP,                                                                \
    PETSC_CONCAT(HIPBLAS,HIPORIGINAL),                                  \
    PETSC_CONCAT(CUPMBLAS,MAPPED)                                       \
  )

// if both cuda and hip agree on the same name
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE(STEM)             \
  PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(STEM,STEM,STEM)

PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE(_STATUS_SUCCESS)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE(_STATUS_NOT_INITIALIZED)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE(_STATUS_ALLOC_FAILED)

} // namespace Impl

} // namespace Petsc
