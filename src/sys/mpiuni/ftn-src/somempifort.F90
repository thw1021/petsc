!
!     Called from C so they should not be in a module
!
      subroutine MPIUNISetModuleBlock()
      use mpiuni
      implicit none
      call MPIUNISetFortranBasePointers(MPI_IN_PLACE)
      end subroutine MPIUNISetModuleBlock
