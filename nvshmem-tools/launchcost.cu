#include <cuda_runtime.h>
#include <stdio.h>
__global__ void empty(void) {}
__global__ void spin(volatile int *f) { while (*f == 0) ; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int main(void){
  const int N=2000; cudaStream_t s,s2; cudaEvent_t e1,e2; double t0,t1;
  cudaSetDevice(0); cudaStreamCreate(&s); cudaStreamCreate(&s2);
  cudaEventCreateWithFlags(&e1,cudaEventDisableTiming); cudaEventCreateWithFlags(&e2,cudaEventDisableTiming);
  for(int i=0;i<100;i++) empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s);

  t0=now(); for(int i=0;i<N;i++) empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s); t1=now();
  printf("empty <<<1,1>>> pipelined      : %6.2f us/launch\n",(t1-t0)/N*1e6);

  t0=now(); for(int i=0;i<N;i++){ empty<<<1,1,0,s>>>(); cudaStreamSynchronize(s);} t1=now();
  printf("empty <<<1,1>>> + stream sync  : %6.2f us/launch\n",(t1-t0)/N*1e6);

  t0=now(); for(int i=0;i<N;i++){ cudaEventRecord(e1,s); cudaStreamWaitEvent(s2,e1,0);
                                  cudaEventRecord(e2,s2); cudaStreamWaitEvent(s,e2,0);} cudaStreamSynchronize(s); t1=now();
  printf("2x(EventRecord+StreamWait)     : %6.2f us/pair-of-pairs\n",(t1-t0)/N*1e6);

  // the shape PETSc actually issues per exchange on the comm stream
  t0=now(); for(int i=0;i<N;i++){ cudaEventRecord(e1,s); cudaStreamWaitEvent(s2,e1,0);
                                  empty<<<1,1,0,s2>>>(); empty<<<1,1,0,s2>>>(); empty<<<1,1,0,s2>>>();
                                  cudaEventRecord(e2,s2); cudaStreamWaitEvent(s,e2,0);
                                  empty<<<1,1,0,s>>>(); empty<<<1,1,0,s>>>(); }
       cudaStreamSynchronize(s); t1=now();
  printf("PETSc-shaped: 5 launches+4 evt : %6.2f us/exchange\n",(t1-t0)/N*1e6);
  return 0; }
