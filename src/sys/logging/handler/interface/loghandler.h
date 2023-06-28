#if !defined(PETSCLOGHANDLER_H)
#define PETSCLOGHANDLER_H

#include <petsc/private/loghandlerimpl.h>

#define PetscLogHandlerTry(h,Name,args) \
  do { \
    PetscLogHandler _h = (h); \
    if (_h && (_h)->Name) PetscCall((*((_h)->Name))args); \
  } while(0)

#endif // #define PETSCLOGHANLDER_H
