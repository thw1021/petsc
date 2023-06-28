#if !defined(PETSCLOGHANDLER_H)
#define PETSCLOGHANDLER_H

#include <petsc/private/loghandlerimpl.h>

#define PetscLogHandlerTry(h,Name,...) \
  do { \
    PetscLogHandler _h = (h); \
    if (_h && (_h)->Name) PetscCall((*((_h)->Name))(h, __VA_ARGS__)); \
  } while(0)

#endif // #define PETSCLOGHANLDER_H
