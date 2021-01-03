cdef extern from "Python.h":
    ctypedef struct PyObject
    void Py_INCREF(PyObject*)
    void Py_DECREF(PyObject*)

cimport cpython
from libc cimport stdlib
from libc.stdint cimport int64_t, uint64_t, uint8_t, uint16_t
from cpython cimport pycapsule
from cpython cimport array

cdef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes

ctypedef struct DLContext:
    int device_type
    int device_id

cdef enum DLDataTypeCode:
    kDLInt = <unsigned int>0
    kDLUInt = <unsigned int>1
    kDLFloat = <unsigned int>2

cdef struct DLTensor:
    size_t data  # Safer than "void *"
    DLContext ctx
    int ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset

cdef struct DLManagedTensor:
    DLTensor dl_tensor
    void* manager_ctx
    void(*deleter)(DLManagedTensor*)

cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor
    try:
        dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(dltensor, 'used_dltensor')
        return             # we do not call a used capsule's deleter
    except Exception:
        dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(dltensor, 'dltensor')
    deleter(dlm_tensor)


cdef void deleter(DLManagedTensor* tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    CHKERR( PetscObjectDereference(<PetscObject>tensor.manager_ctx) )
    stdlib.free(tensor)
    tensor.manager_ctx = NULL
