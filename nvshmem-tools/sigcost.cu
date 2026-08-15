/* Is a host-side stream-ordered NVSHMEM signal op cheaper than a <<<1,1>>> kernel? */
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void empty(void) {}
__global__ void sigset_kernel(uint64_t *s, int pe) { if (!threadIdx.x) nvshmemx_signal_op(s, 1, NVSHMEM_SIGNAL_SET, pe); }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int main(int argc,char**argv){
  const int N=2000; int rank,lr=0,ndev,mype; const char*e; double t0,t1; cudaStream_t s;
  MPI_Comm comm=MPI_COMM_WORLD; nvshmemx_init_attr_t attr=NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Init(&argc,&argv); MPI_Comm_rank(comm,&rank);
  e=getenv("OMPI_COMM_WORLD_LOCAL_RANK"); if(e) lr=atoi(e);
  cudaGetDeviceCount(&ndev); cudaSetDevice(lr%ndev);
  attr.mpi_comm=&comm; nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr);
  mype=nvshmem_my_pe();
  uint64_t *sig=(uint64_t*)nvshmem_calloc(1,sizeof(uint64_t));
  cudaStreamCreate(&s);
  for(int i=0;i<50;i++) empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s);

  t0=now(); for(int i=0;i<N;i++) empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s); t1=now();
  if(!rank) printf("  baseline empty kernel launch          : %6.2f us\n",(t1-t0)/N*1e6);

  t0=now(); for(int i=0;i<N;i++) sigset_kernel<<<1,1,0,s>>>(sig,mype); cudaStreamSynchronize(s); t1=now();
  if(!rank) printf("  <<<1,1>>> kernel doing signal_op      : %6.2f us\n",(t1-t0)/N*1e6);

  t0=now(); for(int i=0;i<N;i++) nvshmemx_signal_op_on_stream(sig,1,NVSHMEM_SIGNAL_SET,mype,s); cudaStreamSynchronize(s); t1=now();
  if(!rank) printf("  nvshmemx_signal_op_on_stream (host)  : %6.2f us\n",(t1-t0)/N*1e6);

  /* condition already true -> measures pure enqueue overhead of the wait */
  t0=now(); for(int i=0;i<N;i++) nvshmemx_signal_wait_until_on_stream(sig,NVSHMEM_CMP_EQ,1,s); cudaStreamSynchronize(s); t1=now();
  if(!rank) printf("  nvshmemx_signal_wait_until_on_stream : %6.2f us\n",(t1-t0)/N*1e6);

  nvshmem_free(sig); nvshmem_finalize(); MPI_Finalize(); return 0; }
