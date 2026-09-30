static const char help[] = "Test CUDA/HIP MatZeroEntries() ordering and MatScale()/MatAXPY() barrier semantics on the current device context.\n";

#include <petscmat.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

static PetscErrorCode WriteValues(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[], PetscBool delay)
{
  PetscScalar *a;
  void        *stream;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCall(MatSeqAIJCUSPARSEGetArray(A, &a));
    if (delay) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCallCUDA(cudaMemcpyAsync(a, input, 4 * sizeof(*a), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(A, &a));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCall(MatSeqAIJHIPSPARSEGetArray(A, &a));
    if (delay) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCallHIP(hipMemcpyAsync(a, input, 4 * sizeof(*a), hipMemcpyHostToDevice, *(hipStream_t *)stream));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(A, &a));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                   A, B;
  PetscDeviceContext    saved, current;
  PetscDevice           device;
  PetscDeviceType       type;
  PetscScalar          *input;
  const PetscScalar    *a;
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(PetscDeviceMalloc(saved, PETSC_MEMTYPE_HOST, 4, &input));
  for (PetscInt i = 0; i < 4; ++i) input[i] = i + 1;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 1, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(WriteValues(A, type, saved, input, PETSC_FALSE));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(PetscDeviceContextSynchronize(saved));

  for (PetscInt k = 0; k < 2; ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &current));
    PetscCall(PetscDeviceContextSetStreamType(current, streams[k]));
    PetscCall(PetscDeviceContextSetUp(current));
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    // Warm the zeroing operation before delaying the input write.
    PetscCall(MatZeroEntries(A));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_TRUE));
    PetscCall(MatZeroEntries(A));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: CSR entry %" PetscInt_FMT " is %g instead of zero", PetscStreamTypes[streams[k]], i, (double)PetscRealPart(a[i]));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

    // Warm the BLAS handle and scaling kernel before checking completion on return.
    PetscCall(MatScale(A, 2));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_TRUE));
    PetscCall(MatScale(A, 2));
    if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(current, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatScale() returned with work pending on a barrier context");
    }
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: incorrectly scaled CSR entry %" PetscInt_FMT, PetscStreamTypes[streams[k]], i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

    // Warm AXPY before delaying the update of its source matrix.
    PetscCall(MatAXPY(A, 2, B, SAME_NONZERO_PATTERN));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_FALSE));
    PetscCall(WriteValues(B, type, current, input, PETSC_TRUE));
    PetscCall(MatAXPY(A, 2, B, SAME_NONZERO_PATTERN));
    if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(current, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatAXPY() returned with work pending on a barrier context");
    }
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 3 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: incorrect CSR entry %" PetscInt_FMT " after MatAXPY()", PetscStreamTypes[streams[k]], i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&current));
  }

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscDeviceFree(saved, input));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out

    test:
      suffix: cuda
      requires: cuda

    test:
      suffix: hip
      requires: hip

TEST*/
