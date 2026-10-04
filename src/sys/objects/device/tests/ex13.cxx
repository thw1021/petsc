static const char help[] = "Test stream ordering of CUPM two-dimensional memory operations.\n";

#include <petsc/private/cupminterface.hpp>
#include <chrono>
#include <thread>

using Petsc::device::cupm::DeviceType;

template <DeviceType T>
struct TestMemory : Petsc::device::cupm::impl::Interface<T> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  static void Delay(void *) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

  static PetscErrorCode Run(PetscBool test_memset)
  {
    const std::size_t width = 3, height = 2, src_pitch = 7, dst_pitch = 5;
    const auto        stream_flags = cupmStreamNonBlocking;
    const auto        dtod = cupmMemcpyDeviceToDevice, dtoh = cupmMemcpyDeviceToHost;
    unsigned char    *src, *dst;
    unsigned char     result[10];
    cupmStream_t      stream;

    PetscFunctionBeginUser;
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
    PetscCallCUPM(cupmStreamCreateWithFlags(&stream, stream_flags));
    PetscCallCUPM(cupmMalloc(reinterpret_cast<void **>(&src), src_pitch * height));
    PetscCallCUPM(cupmMalloc(reinterpret_cast<void **>(&dst), dst_pitch * height));
    PetscCallCUPM(cupmMemset(src, 1, src_pitch * height));
    PetscCallCUPM(cupmMemset(dst, 7, dst_pitch * height));
    PetscCallCUPM(cupmDeviceSynchronize());
    PetscCallCUPM(cupmLaunchHostFunc(stream, Delay, nullptr));
    if (test_memset) {
      PetscCallCUPM(cupmMemsetAsync(dst, 2, dst_pitch * height, stream));
      PetscCallCUPM(cupmMemset2DAsync(dst, dst_pitch, 0, width, height, stream));
    } else {
      PetscCallCUPM(cupmMemsetAsync(src, 2, src_pitch * height, stream));
      PetscCallCUPM(cupmMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, dtod, stream));
    }
    // Finish any work accidentally submitted to the default stream before the delayed producer.
    PetscCallCUPM(cupmStreamSynchronize(nullptr));
    PetscCallCUPM(cupmStreamSynchronize(stream));
    PetscCallCUPM(cupmMemcpy(result, dst, sizeof(result), dtoh));
    for (std::size_t j = 0; j < height; ++j) {
      for (std::size_t i = 0; i < dst_pitch; ++i) {
        const unsigned char expected = i < width ? (test_memset ? 0 : 2) : (test_memset ? 2 : 7);

        PetscCheck(result[j * dst_pitch + i] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect %s result at row %zu, column %zu: got %u, expected %u", test_memset ? "memset" : "copy", j, i, (unsigned int)result[j * dst_pitch + i], (unsigned int)expected);
      }
    }
    PetscCallCUPM(cupmFree(src));
    PetscCallCUPM(cupmFree(dst));
    PetscCallCUPM(cupmStreamDestroy(stream));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

int main(int argc, char **argv)
{
  PetscBool          test_memset = PETSC_FALSE;
  PetscDeviceContext dctx;
  PetscDeviceType    dtype;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-test_memset", &test_memset, nullptr));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  switch (dtype) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(TestMemory<DeviceType::CUDA>::Run(test_memset));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(TestMemory<DeviceType::HIP>::Run(test_memset));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "This test requires CUDA or HIP");
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cxx

  testset:
    output_file: output/empty.out
    args: -test_memset {{0 1}separate output}
    test:
      suffix: cuda
      requires: cuda
      args: -default_device_type cuda
    test:
      suffix: hip
      requires: hip
      args: -default_device_type hip

TEST*/
