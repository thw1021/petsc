/* Bare GPU-aware MPI ping-pong on plain cudaMalloc buffers. */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
  int rank,size,ndev,lr=0; const char*e; double *buf; int n=1024;
  MPI_Init(&argc,&argv); MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  e=getenv("OMPI_COMM_WORLD_LOCAL_RANK"); if(e) lr=atoi(e);
  cudaGetDeviceCount(&ndev); cudaSetDevice(lr%ndev);
  if(cudaMalloc((void**)&buf,n*sizeof(double))!=cudaSuccess){printf("cudaMalloc failed\n");MPI_Abort(MPI_COMM_WORLD,1);}
  cudaMemset(buf,0,n*sizeof(double));
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0){ MPI_Send(buf,n,MPI_DOUBLE,1,0,MPI_COMM_WORLD);
               MPI_Recv(buf,n,MPI_DOUBLE,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
  else if(rank==1){ MPI_Recv(buf,n,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
               MPI_Send(buf,n,MPI_DOUBLE,0,0,MPI_COMM_WORLD); }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0) printf("bare GPU-aware MPI ping-pong: OK\n");
  cudaFree(buf); MPI_Finalize(); return 0; }
