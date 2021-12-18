!
!
      subroutine MPIUNISetModuleBlock()
      use mpi
      implicit none
      call MPIUNISetFortranBasePointers(MPI_IN_PLACE)
      return
      end
