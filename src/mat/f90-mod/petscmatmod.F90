#include <petsc/finclude/petscmat.h>

        module petscmat
        use petscmatdef
        use petscvec
#include <../src/mat/f90-mod/petscmat.h90>
        interface
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h90>
        end interface
        end module

