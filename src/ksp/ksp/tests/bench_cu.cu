#include <cuda_profiler_api.h>
#define cupmDeviceSynchronize cudaDeviceSynchronize
#define cupmProfilerStart     cudaProfilerStart
#define cupmProfilerStop      cudaProfilerStop
#define PetscCallCUPM         PetscCallCUDA

#include "bench_impl.hpp"
