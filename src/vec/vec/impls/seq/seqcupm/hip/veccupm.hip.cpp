#include "../veccupm.hpp"

using namespace Petsc;

static constexpr auto VecSeqHIP = Impl::VecSeq_CUPM<CUPMDeviceType::HIP>();
