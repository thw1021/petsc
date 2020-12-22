module riemannsolver

    use petscsys
#include <petsc/finclude/petscsys.h>

    use input_parameter

    implicit none

    contains

    subroutine RSChoice(dim, Nf, qp, n, uL, uR, numConstants, constants, flux, ctx, ierr)

        PetscErrorCode, intent(out)     :: ierr
        PetscInt                        :: dim, Nf, numConstants
        PetscReal, dimension(0:ndim-1)  :: qp, n
        PetscScalar, dimension(0:nvar-1):: uL, uR, constants, flux
        PetscScalar, dimension(:)       :: ctx

        PetscReal, dimension(0:ndim-1)  :: nn
        PetscReal                       :: norm
        PetscInt                        :: idim

        ! Normalize the normal n
        norm = 0.d0
        do idim = 0, ndim-1
            nn(idim) = n(idim)
            norm     = norm + nn(idim)*nn(idim)
        end do
        norm = sqrt(norm)
        nn   = nn/norm

        ! Call user-chosen Riemann solver
        if (RSname == "HLL") then
            call HLL(uL, uR, nn, flux)
        end if

        ! Weight the flux by the area of the face
        flux = flux * norm

    end subroutine RSChoice

    subroutine HLL(consL, consR, nn, flux)

        PetscScalar, dimension(0:nvar-1)              :: consL, consR
        PetscReal, dimension(0:ndim-1)                :: nn
        PetscScalar, dimension(0:nvar-1), intent(out) :: flux
        PetscReal                                     :: rhoL, uL, vL, wL, pL, unL, aL, SL
        PetscReal                                     :: rhoR, uR, vR, wR, pR, unR, aR, SR
        PetscReal, dimension(0:nvar-1)                :: fL, fR

        rhoL  = consL(0)
        uL    = consL(1)/rhoL
        vL    = consL(2)/rhoL
        wL    = consL(3)/rhoL
        pL    = (gamma-1)*(consL(4)-rhoL*(uL**2 + vL**2 + wL**2)/2)
        unL   = uL*nn(0) + vL*nn(1) + wL*nn(2)
        aL    = sqrt((gamma*pL)/rhoL)

        rhoR  = consR(0)
        uR    = consR(1)/rhoR
        vR    = consR(2)/rhoR
        wR    = consR(3)/rhoR
        pR    = (gamma-1)*(consR(4)-rhoR*(uR**2 + vR**2 + wR**2)/2)
        unR   = uR*nn(0) + vR*nn(1) + wR*nn(2)
        aR    = sqrt((gamma*pR)/rhoR)

        SR    = max(abs(unR-aR), abs(unL-aL), abs(unR+aR), abs(unL+aL))
        SL    = -SR

        ! PhysicaL left flux
        fL(0) = rhoL*unL
        fL(1) = rhoL*unL*uL + pL*nn(0)
        fL(2) = rhoL*unL*vL + pL*nn(1)
        fL(3) = rhoL*unL*wL + pL*nn(2)
        fL(4) = unL*(consL(4)+pL)
        ! Physical right flux
        fR(0) = rhoR*unR
        fR(1) = rhoR*unR*uR + pR*nn(0)
        fR(2) = rhoR*unR*vR + pR*nn(1)
        fR(3) = rhoR*unR*wR + pR*nn(2)
        fR(4) = unR*(consR(4)+pR)

        ! HLL flux
        flux = 1.d0/(SR-SL)*(SR*fL - SL*fR + SL*SR*(consR-consL))

    end subroutine HLL

end module riemannsolver