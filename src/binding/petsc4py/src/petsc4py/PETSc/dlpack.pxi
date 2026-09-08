# DLPack interface

cdef extern from "Python.h":
    ctypedef void (*PyCapsule_Destructor)(object)
    bint PyCapsule_IsValid(object, const char*)
    void* PyCapsule_GetPointer(object, const char*) except? NULL
    int PyCapsule_SetName(object, const char*) except -1
    int PyCapsule_SetDestructor(object, PyCapsule_Destructor) except -1
    object PyCapsule_New(void*, const char*, PyCapsule_Destructor)
    int PyCapsule_CheckExact(object)

cdef extern from "<stdlib.h>" nogil:
    ctypedef signed long int64_t
    ctypedef unsigned long long uint64_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned short uint16_t
    void free(void* ptr)
    void* malloc(size_t size)

cdef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes

cdef enum PetscDLDeviceType:
    kDLCPU = <unsigned int>1
    kDLCUDA = <unsigned int>2
    kDLCUDAHost = <unsigned int>3
    # kDLOpenCL = <unsigned int>4
    # kDLVulkan = <unsigned int>7
    # kDLMetal = <unsigned int>8
    # kDLVPI = <unsigned int>9
    kDLROCM = <unsigned int>10
    kDLROCMHost = <unsigned int>11
    # kDLExtDev = <unsigned int>12
    kDLCUDAManaged = <unsigned int>13
    # kDLOneAPI = <unsigned int>14

ctypedef struct DLContext:
    PetscDLDeviceType device_type
    int device_id

cdef enum DLDataTypeCode:
    kDLInt = <unsigned int>0
    kDLUInt = <unsigned int>1
    kDLFloat = <unsigned int>2
    kDLComplex = <unsigned int>5

cdef struct DLTensor:
    void* data
    DLContext ctx
    int ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset

cdef inline void dlpack_dtype(DLDataType* dtype) except *:
    if 8 * sizeof(PetscScalar) > 255:
        raise ValueError('Unsupported PetscScalar type')
    if sizeof(PetscScalar) == sizeof(PetscReal):
        dtype.code = <uint8_t>DLDataTypeCode.kDLFloat
    elif sizeof(PetscScalar) == 2 * sizeof(PetscReal):
        dtype.code = <uint8_t>DLDataTypeCode.kDLComplex
    else:
        raise ValueError('Unsupported PetscScalar type')
    dtype.bits = <uint8_t>(8 * sizeof(PetscScalar))
    dtype.lanes = <uint16_t>1

cdef inline void dlpack_validate_dtype(DLDataType* dtype) except *:
    cdef uint8_t code = <uint8_t>DLDataTypeCode.kDLFloat

    if sizeof(PetscScalar) == 2 * sizeof(PetscReal):
        code = <uint8_t>DLDataTypeCode.kDLComplex
    if dtype.code != code or dtype.bits != 8 * sizeof(PetscScalar) or dtype.lanes != 1:
        raise TypeError('DLPack tensor dtype does not match PETSc ScalarType')

cdef inline bint dlpack_stream_sync_needed(PetscDLDeviceType device_type, object stream) except -1:
    if device_type in (kDLCPU, kDLCUDAHost, kDLROCMHost):
        if stream is not None:
            raise RuntimeError('DLPack on host memory only supports stream=None')
        return False
    if device_type not in (kDLCUDA, kDLCUDAManaged, kDLROCM):
        raise BufferError('Unsupported DLPack device type')
    if stream is None:
        return True
    if not hasattr(stream, '__index__'):
        raise TypeError('DLPack stream must be an integer or None')
    stream = stream.__index__()
    if stream < -1:
        raise ValueError('DLPack stream must be greater than or equal to -1')
    if device_type in (kDLCUDA, kDLCUDAManaged) and stream == 0:
        raise ValueError('DLPack stream=0 is invalid for CUDA')
    if device_type == kDLROCM and stream in (1, 2):
        raise ValueError('DLPack stream=1 and stream=2 are invalid for ROCm')
    return stream != -1

ctypedef int (*dlpack_manager_del_obj)(void*) noexcept nogil

cdef struct DLManagedTensor:
    DLTensor dl_tensor
    void* manager_ctx
    void (*manager_deleter)(DLManagedTensor*) noexcept nogil
    dlpack_manager_del_obj del_obj

cdef void pycapsule_deleter(object dltensor) noexcept:
    cdef DLManagedTensor* dlm_tensor = NULL
    # we do not call a used capsule's deleter
    if PyCapsule_IsValid(dltensor, b'dltensor'):
        dlm_tensor = <DLManagedTensor *>PyCapsule_GetPointer(dltensor, b'dltensor')
        manager_deleter(dlm_tensor)

cdef void used_pycapsule_deleter(object dltensor) noexcept:
    cdef DLManagedTensor* dlm_tensor = NULL
    if PyCapsule_IsValid(dltensor, b'used_dltensor'):
        dlm_tensor = <DLManagedTensor *>PyCapsule_GetPointer(dltensor, b'used_dltensor')
        if dlm_tensor.manager_deleter != NULL:
            dlm_tensor.manager_deleter(dlm_tensor)

cdef void manager_deleter(DLManagedTensor* tensor) noexcept nogil:
    if tensor.manager_ctx is NULL:
        return
    free(tensor.dl_tensor.shape)
    if tensor.del_obj is not NULL:
        tensor.del_obj(&tensor.manager_ctx)
    free(tensor)

# --------------------------------------------------------------------
