#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,1);} }while(0)
__global__ void send_kernel(int *dst,const int *src,uint64_t *sig,int peer,int n){
  if(threadIdx.x==0&&blockIdx.x==0){ nvshmem_int_put(dst,src,n,peer); nvshmem_fence();
    nvshmemx_signal_op(sig,1,NVSHMEM_SIGNAL_SET,peer);} }
__global__ void wait_kernel(uint64_t *sig){
  if(threadIdx.x==0&&blockIdx.x==0) nvshmem_signal_wait_until(sig,NVSHMEM_CMP_EQ,1); }
int main(int argc,char**argv){
  const int n=1024; int rank,size,local_rank=0,ndev,mype,npes,mype_node,errs=0;
  const char*lr; MPI_Comm comm=MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr=NVSHMEMX_INIT_ATTR_INITIALIZER;
  int *dst,*src_h,*src_d; uint64_t *sig; cudaDeviceProp prop;
  MPI_Init(&argc,&argv); MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);
  lr=getenv("OMPI_COMM_WORLD_LOCAL_RANK"); if(lr) local_rank=atoi(lr);
  CUDA_CHK(cudaGetDeviceCount(&ndev));
  CUDA_CHK(cudaSetDevice(local_rank%ndev));
  CUDA_CHK(cudaGetDeviceProperties(&prop,local_rank%ndev));
  attr.mpi_comm=&comm;
  if(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr)){fprintf(stderr,"init failed\n");MPI_Abort(comm,1);}
  mype=nvshmem_my_pe(); npes=nvshmem_n_pes(); mype_node=nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("rank %d/%d -> PE %d/%d (node PE %d) cuda dev %d bus 0x%02x\n",
         rank,size,mype,npes,mype_node,local_rank%ndev,prop.pciBusID); fflush(stdout);
  dst=(int*)nvshmem_malloc(n*sizeof(int)); sig=(uint64_t*)nvshmem_calloc(1,sizeof(uint64_t));
  if(!dst||!sig){fprintf(stderr,"nvshmem_malloc failed\n");MPI_Abort(comm,1);}
  src_h=(int*)malloc(n*sizeof(int)); for(int i=0;i<n;i++) src_h[i]=1000*mype+i;
  /* The put SOURCE must also live on the symmetric heap (or be nvshmemx_buffer_register()ed):
     the IB transports need an lkey for it, and an unregistered cudaMalloc source dies with
     IBV_WC_LOC_PROT_ERR (status 4) in the proxy the first time a put crosses a node. */
  src_d=(int*)nvshmem_malloc(n*sizeof(int)); if(!src_d){fprintf(stderr,"nvshmem_malloc src failed\n");MPI_Abort(comm,1);}
  CUDA_CHK(cudaMemcpy(src_d,src_h,n*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemset(dst,0,n*sizeof(int)));
  nvshmem_barrier_all();
  send_kernel<<<1,32>>>(dst,src_d,sig,(mype+1)%npes,n); CUDA_CHK(cudaGetLastError());
  wait_kernel<<<1,32>>>(sig); CUDA_CHK(cudaGetLastError()); CUDA_CHK(cudaDeviceSynchronize());
  { int *got=(int*)malloc(n*sizeof(int)); int from=(mype-1+npes)%npes;
    CUDA_CHK(cudaMemcpy(got,dst,n*sizeof(int),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) if(got[i]!=1000*from+i){ if(errs<3) fprintf(stderr,"PE %d dst[%d]=%d want %d\n",mype,i,got[i],1000*from+i); errs++; }
    free(got); }
  printf("PE %d: put+signal from PE %d -> %s (%d mismatches); my target PE %d is %s\n",mype,(mype-1+npes)%npes,errs?"FAIL":"OK",errs,(mype+1)%npes,nvshmem_ptr(dst,(mype+1)%npes)?"P2P-mapped":"REMOTE (IB)");
  fflush(stdout);
  nvshmem_barrier_all(); free(src_h); nvshmem_free(src_d);
  nvshmem_free(dst); nvshmem_free(sig); nvshmem_finalize(); MPI_Finalize();
  return errs?1:0; }
