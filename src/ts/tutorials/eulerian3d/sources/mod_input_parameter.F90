module input_parameter

    implicit none

    !> Debug
    logical, parameter                :: debug              = .true.
    !> @var Max string length
    integer, parameter                :: MSTRLEN            = 999
    !> @var Name of the mesh to read
    character(len=MSTRLEN), parameter :: meshname           = "cube.msh"
    ! Number of DOF (5 for 3D)
    integer, parameter                :: nvar               = 5
    ! Parameter for dimension
    integer, parameter                :: ndim               = 3
    !> @var Ratio of specific heats
    double precision,  parameter      :: gamma              = 1.4d0
    !> @var Cell ID to decide spatial discretization order  (0-P0(1st Order), 1-P1(2nd Order))
    integer, parameter                :: IDinit             = 0
    !> @var Name of the Riemann solver used to compute the
    !> fluxes at the cells interfaces (roe, HLL,RHLL, HLLC, HLLE, Rusanov, GGcell)
    character(len=MSTRLEN), parameter :: RSname             = "HLL"
    !> @var Courant number for the explicit temporal integration
    double precision, parameter       :: CFL                = 0.7d0
    !> @var Final time to be reached by temporal integration
    double precision, parameter       :: tfinal             = 1.0d0
    !> @var Maximum number of iterations of the time loop
    !> The computation stops either at tfinal or at maxiter,
    !> whichever is the first condition met
    integer, parameter                :: maxiter            = 1000000
    !> @var Order of the Runge-Kutta integration
    !> RKorder = 1 : Explicit Euler
    !> RKorder = 2 : RK 2 SSP
    !> RKorder = 3 : RK 3 SSP
    !> RKorder = 4 : RK 4 SSP
    integer, parameter                :: RKorder            = 3
    !> @var Number of steps for the Runge-Kutta integration
    !> RKorder = 1 : RKstep is ignored (number of stages = 1)
    !> RKorder = 2 : number of stages = RKstep
    !> RKorder = 3 : number of stages = RKstep * RKstep
    !> RKorder = 4 : RKstep is ignored (number of stages = 10)
    !> Documentation at https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSSP.html
    !> See also :
    !> Ketcheson, Highly efficient strong stability preserving Runge Kutta methods with low storage implementations, SISC, 2008
    !> Gottlieb, Ketcheson, and Shu, High order strong stability preserving time discretizations, J Scientific Computing, 2009
    integer, parameter                :: RKstep             = 2
    !> @var Frequency, in iterations, to write info on screen
    integer, parameter                :: screen_output_freq = 1
    !> @var Frequency, in iterations, to dump a solution file
    integer, parameter                :: file_output_freq   = 10
    !> @var Name of the case to initialize 
    character(len=MSTRLEN), parameter :: case               = "sodx"

end module input_parameter