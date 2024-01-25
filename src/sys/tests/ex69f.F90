    program ex69F90

!   Demonstrates how using mpiexec to start up a program can dramatically change
!   the OpenMP thread binding/mapping resulting in poor performance

!   Set the environmental variable with, for example,
!      export OMP_NUM_THREADS=4
!   Run this example on one MPI process three ways
!      ./ex69f
!      mpiexec -n 1 ./ex69f
!      mpiexec --bind-to numa -n 1 ./ex69f
!
!   You may get very different wall clock times
!   It seems some mpiexec implementations change the thred binding/mapping that results with
!   OpenMP so all the threads are run on a single core
!
!   The same differences occur without the PetscInitialize() call indicating
!   the binding change is done by the mpiexec, not the MPI_Init()

#include <petsc/finclude/petscsys.h>
    use petsc
    implicit none

    PetscErrorCode ierr
    double precision cputime_start,cputime_end,wtime_start,wtime_end,omp_get_wtime
    integer(kind = 8) systime_start,systime_end,systime_rate
    double precision x(10000000)
    integer i,maxthreads,omp_get_max_threads

    PetscCallA(PetscInitialize(ierr))
    call cpu_time(cputime_start)
    call system_clock(systime_start,systime_rate)
    wtime_start = omp_get_wtime()
!$OMP PARALLEL DO
    do i=1,10000000
      x(i) = exp(3.0d0*i)
    enddo
    call cpu_time(cputime_end)
    call system_clock(systime_end,systime_rate)
    wtime_end = omp_get_wtime()
    print*,'CPU time reported by cpu_time()            ', cputime_end - cputime_start
    print*,'Wall clock time reported by system_clock() ',real(systime_end - systime_start,kind=8)/real(systime_rate,kind=8)
    print*,'Wall clock time reported by omp_get_wtime()', wtime_end - wtime_start
    print*,'Value of x(22)',x(22)
!$  maxthreads = omp_get_max_threads()
    print*,'Number of threads set',maxthreads
    PetscCallA(PetscFinalize(ierr))
end program ex69F90

!/*TEST
!
!   build:
!     requires: openmp
!
!   test:
!
!TEST*/
