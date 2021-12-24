#ifndef __MPIIMPL_H
#define __MPIIMPL_H

#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscGatherNumberOfMessages_Private(MPI_Comm,const PetscMPIInt[],const PetscInt[],PetscMPIInt*);
PETSC_EXTERN PetscErrorCode PetscGatherMessageLengths_Private(MPI_Comm,PetscMPIInt,PetscMPIInt,const PetscInt[],PetscMPIInt**,PetscInt**);

#if !defined(PETSC_HAVE_MPI_LARGE_COUNT) /* No matter PetscInt is 32-bit or 64-bit */
  PETSC_STATIC_INLINE PetscErrorCode MPIU_Send(const void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt dest,PetscMPIInt tag,MPI_Comm comm)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Send(buf,count2,datatype,dest,tag,comm);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PetscErrorCode MPIU_Send_init(const void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt dest,PetscMPIInt tag,MPI_Comm comm,MPI_Request *request)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Send_init(buf,count2,datatype,dest,tag,comm,request);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PetscErrorCode MPIU_Isend(const void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt dest,PetscMPIInt tag,MPI_Comm comm,MPI_Request *request)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Isend(buf,count2,datatype,dest,tag,comm,request);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PetscErrorCode MPIU_Recv(void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt source,PetscMPIInt tag,MPI_Comm comm,MPI_Status *status)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Recv(buf,count2,datatype,source,tag,comm,status);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PetscErrorCode MPIU_Recv_init(void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt source,PetscMPIInt tag,MPI_Comm comm,MPI_Request* request)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Recv_init(buf,count2,datatype,source,tag,comm,request);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PetscErrorCode MPIU_Irecv(void *buf,PetscInt count,MPI_Datatype datatype,PetscMPIInt source,PetscMPIInt tag,MPI_Comm comm,MPI_Request* request)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Irecv(buf,count2,datatype,source,tag,comm,request);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }
 #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  PETSC_STATIC_INLINE PetscErrorCode MPIU_Reduce_local(const void *inbuf,void *inoutbuf,PetscInt count,MPI_Datatype datatype,MPI_Op op)
  {
    PetscErrorCode ierr;
    PetscMPIInt    count2;

    PetscFunctionBegin;
    ierr = PetscMPIIntCast(count,&count2);CHKERRQ(ierr);
    ierr = MPI_Reduce_local(inbuf,inoutbuf,count,datatype,op);CHKERRMPI(ierr);
    PetscFunctionReturn(0);
  }
 #endif

#elif defined(PETSC_USE_64BIT_INDICES)
  #define MPIU_Send(buf,count,datatype,dest,tag,comm)                         MPI_Send_c(buf,count,datatype,dest,tag,comm)
  #define MPIU_Send_init(buf,count,datatype,dest,tag,comm,request)            MPI_Send_init_c(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Isend(buf,count,datatype,dest,tag,comm,request)                MPI_Isend_c(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Recv(buf,count,datatype,source,tag,comm,status)                MPI_Recv_c(buf,count,datatype,source,tag,comm,status)
  #define MPIU_Recv_init(buf,count,datatype,source,tag,comm,request)          MPI_Recv_init_c(buf,count,datatype,source,tag,comm,request)
  #define MPIU_Irecv(buf,count,datatype,source,tag,comm,request)              MPI_Irecv_c(buf,count,datatype,source,tag,comm,request)
 #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  #define MPIU_Reduce_local(inbuf,inoutbuf,count,datatype,op)                 MPI_Reduce_local_c(inbuf,inoutbuf,count,datatype,op)
 #endif
#else
  #define MPIU_Send(buf,count,datatype,dest,tag,comm)                         MPI_Send(buf,count,datatype,dest,tag,comm)
  #define MPIU_Send_init(buf,count,datatype,dest,tag,comm,request)            MPI_Send_init(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Isend(buf,count,datatype,dest,tag,comm,request)                MPI_Isend(buf,count,datatype,dest,tag,comm,request)
  #define MPIU_Recv(buf,count,datatype,source,tag,comm,status)                MPI_Recv(buf,count,datatype,source,tag,comm,status)
  #define MPIU_Recv_init(buf,count,datatype,source,tag,comm,request)          MPI_Recv_init(buf,count,datatype,source,tag,comm,request)
  #define MPIU_Irecv(buf,count,datatype,source,tag,comm,request)              MPI_Irecv(buf,count,datatype,source,tag,comm,request)
 #if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  #define MPIU_Reduce_local(inbuf,inoutbuf,count,datatype,op)                 MPI_Reduce_local(inbuf,inoutbuf,count,datatype,op)
 #endif
#endif

#endif
