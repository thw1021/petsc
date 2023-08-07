#define cupmDeviceSynchronize hipDeviceSynchronize
#define cupmProfilerStart     [] { return hipSuccess; }
#define cupmProfilerStop      [] { return hipSuccess; }
#define PetscCallCUPM         PetscCallHIP

#include "bench_impl.hpp"
