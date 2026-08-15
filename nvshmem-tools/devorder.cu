#include <cuda_runtime.h>
#include <stdio.h>
int main(void){int n=0;cudaError_t e=cudaGetDeviceCount(&n);
printf("cudaGetDeviceCount -> %d (%s)\n",n,cudaGetErrorString(e));
for(int i=0;i<n;i++){cudaDeviceProp p;cudaGetDeviceProperties(&p,i);size_t f=0,t=0;
if(cudaSetDevice(i)==cudaSuccess&&cudaMemGetInfo(&f,&t)==cudaSuccess)
printf("  dev %d bus 0x%02x free %.1f/%.1f GiB OK\n",i,p.pciBusID,f/1073741824.0,t/1073741824.0);
else printf("  dev %d bus 0x%02x UNUSABLE\n",i,p.pciBusID);}return 0;}
