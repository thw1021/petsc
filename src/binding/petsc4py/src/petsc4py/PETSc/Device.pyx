# --------------------------------------------------------------------

class staticproperty(property):
  def __get__(self, *args, **kwargs):
    return self.fget.__get__(*args, **kwargs)()

cdef object make_enum_class(str class_name, tuple args):
  cdef dict enum2str = {}
  cdef dict attrs    = {}

  for name, c_enum in args:
    enum2str[c_enum] = name
    attrs[name]      = c_enum

  attrs['__enum2str'] = enum2str
  return type(class_name, (object, ), attrs)

DeviceType = make_enum_class(
  "DeviceType",
  (
    ("HOST"    , PETSC_DEVICE_HOST),
    ("CUDA"    , PETSC_DEVICE_CUDA),
    ("HIP"     , PETSC_DEVICE_HIP),
    ("SYCL"    , PETSC_DEVICE_SYCL),
    ("DEFAULT" , staticproperty(lambda *_,**__: PETSC_DEVICE_DEFAULT()))
  )
)

StreamType = make_enum_class(
  "StreamType",
  (
    ("GLOBAL_BLOCKING"    , PETSC_STREAM_GLOBAL_BLOCKING),
    ("DEFAULT_BLOCKING"   , PETSC_STREAM_DEFAULT_BLOCKING),
    ("GLOBAL_NONBLOCKING" , PETSC_STREAM_GLOBAL_NONBLOCKING),
  )
)

DeviceJoinMode = make_enum_class(
  "DeviceJoinMode",
  (
    ("DESTROY" , PETSC_DEVICE_CONTEXT_JOIN_DESTROY),
    ("SYNC"    , PETSC_DEVICE_CONTEXT_JOIN_SYNC),
    ("NO_SYNC" , PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC),
  )
)

# --------------------------------------------------------------------

cdef class Device:
  """Device object

  Represents a handle to an accelerator (which may be the host)

  See Also
  --------
  DeviceContext, petsc.PetscDevice

  """

  Type = DeviceType

  def __cinit__(self):
    self.device = NULL

  def __dealloc__(self):
    self.destroy()

  @classmethod
  def create(cls, dtype: Type | None = None, device_id: int = DECIDE) -> Device:
    """Create a `Device`

    Not Collective

    Parameters
    ----------
    dtype
        The type of device to create (or `None` for `Device.DEFAULT`)

    device_id
        The numeric id of the device to create

    See Also
    --------
    destroy, petsc.PetscDeviceCreate

    """
    cdef PetscInt        cdevice_id   = asInt(device_id)
    cdef PetscDeviceType cdevice_type = asDeviceType(dtype if dtype is not None else cls.Type.DEFAULT)
    cdef Device          device       = cls()

    CHKERR(PetscDeviceCreate(cdevice_type, cdevice_id, &device.device))
    return device

  def destroy(self) -> None:
    """Destroy a `Device`

    Not Collective

    See Also
    --------
    create, petsc.PetscDeviceDestroy

    """
    CHKERR(PetscDeviceDestroy(&self.device))

  def configure(self) -> None:
    """Configure and setup a `Device`

    Not Collective

    See Also
    --------
    create, petsc.PetscDeviceConfigure

    """
    CHKERR(PetscDeviceConfigure(self.device))

  def view(self, Viewer viewer=None) -> None:
    """View a `Device`

    Collective

    Parameters
    ----------
    viewer
        A `Viewer` instance or `None` for the default viewer

    See Also
    --------
    petsc.PetscDeviceView

    """
    cdef PetscViewer vwr = NULL

    if viewer is not None:
      vwr = viewer.vwr
    CHKERR(PetscDeviceView(self.device, vwr))

  def getDeviceType(self) -> str:
    """Return the `Type` of the `Device`

    Not Collective

    Notes
    -----
    Reading the ``type`` instance property has the same effect as calling this routine

    See Also
    --------
    type, petsc.PetscDeviceGetType

    """
    cdef PetscDeviceType cdtype

    CHKERR(PetscDeviceGetType(self.device, &cdtype))
    return toDeviceType(cdtype)

  property type:
  """The `Type`"""
    def __get__(self) -> str:
      return self.getDeviceType()

  def getDeviceId(self) -> int:
    """Return the device id of the `Device`

    Not Collective

    Notes
    -----
    Reading the ``device_id`` instance property has the same effect as calling this routine

    See Also
    --------
    device_id, create, petsc.PetscDeviceGetDeviceId

    """
    cdef PetscInt cdevice_id = 0

    CHKERR(PetscDeviceGetDeviceId(self.device, &cdevice_id))
    return toInt(cdevice_id)

  property device_id:
  """The device ID"""
    def __get__(self) -> int:
      return self.getDeviceId()

  @staticmethod
  def setDefaultType(device_type: Type | str) -> None:
    """Set the `Type` to be used as the default in subsequent calls to ``create``

    See Also
    --------
    create, petsc.PetscDeviceSetDefaultDeviceType
    """
    cdef PetscDeviceType cdevice_type = asDeviceType(device_type)

    CHKERR(PetscDeviceSetDefaultDeviceType(cdevice_type))

# --------------------------------------------------------------------

cdef class DeviceContext(Object):
  """DeviceContext object

  Represents an abstract handle to a device context

  See Also
  --------
  Device, petsc.PetscDeviceContext

  """
  JoinMode   = DeviceJoinMode
  StreamType = StreamType

  def __cinit__(self):
    self.obj  = <PetscObject*> &self.dctx
    self.dctx = NULL

  def __dealloc__(self):
    self.destroy()

  @classmethod
  def create(cls) -> DeviceContext:
    """Create an empty `DeviceContext`

    Not Collective

    See Also
    --------
    Device, petsc.PetscDeviceContextCreate

    """
    cdef DeviceContext dctx = cls()

    CHKERR(PetscDeviceContextCreate(&dctx.dctx))
    return dctx

  def getStreamType(self) -> str:
    """Return the `DeviceContext.StreamType` of the `DeviceContext`

    Not Collective

    Notes
    -----
    Reading the ``stream_id`` instance property has the same effect as calling this routine

    See Also
    --------
    stream_type, setStreamType, petsc.PetscDeviceContextGetStreamType

    """
    cdef PetscStreamType cstream_type = PETSC_STREAM_DEFAULT_BLOCKING

    CHKERR(PetscDeviceContextGetStreamType(self.dctx, &cstream_type))
    return toStreamType(cstream_type)

  def setStreamType(self, stream_type: StreamType | str) -> None:
    """Set the `DeviceContext.StreamType` of the `DeviceContext`

    Not Collective

    Parameters
    ----------
    stream_type
        The type of stream to set

    Notes
    -----
    Writing the `stream_id` instance property has the same effect as calling this routine

    See Also
    --------
    stream_type, getStreamType, petsc.PetscDeviceContextSetStreamType

    """
    cdef PetscStreamType cstream_type = asStreamType(stream_type)

    CHKERR(PetscDeviceContextSetStreamType(self.dctx, cstream_type))

  property stream_type:
  """The `StreamType`"""
    def __get__(self) -> str:
      return self.getStreamType()

    def __set__(self, stype: StreamType | str) -> None:
      self.setStreamType(stype)

  def getDevice(self) -> Device:
    """Get the `Device` which this `DeviceContext` is attached to

    Not Collective

    Notes
    -----
    Reading the ``device`` instance property has the same effect as calling this routine

    See Also
    --------
    setDevice, device, Device, petsc.PetscDeviceContextGetDevice

    """
    cdef PetscDevice device = NULL

    CHKERR(PetscDeviceContextGetDevice(self.dctx, &device))
    return PyPetscDevice_New(device)

  def setDevice(self, Device device not None) -> None:
    """Set the `Device` which this `DeviceContext` is attached to

    Collective

    Parameters
    ----------
    device
        The `Device` to which this `DeviceContext` is attached to

    Notes
    -----
    Writing the ``device`` instance property has the same effect as calling this routine

    See Also
    --------
    getDevice, device, Device, petsc.PetscDeviceContextSetDevice

    """
    cdef PetscDevice cdevice = PyPetscDevice_Get(device)

    CHKERR(PetscDeviceContextSetDevice(self.dctx, cdevice))

  property device:
  """The `Device`"""
    def __get__(self) -> Device:
      return self.getDevice()

    def __set__(self, Device device) -> None:
      self.setDevice(device)

  def setUp(self) -> None:
    """Set up the internal data structures for using the `DeviceContext`

    Not Collective

    See Also
    --------
    create, destroy, petsc.PetscDeviceContextSetUp

    """
    CHKERR(PetscDeviceContextSetUp(self.dctx))

  def duplicate(self) -> DeviceContext:
    """Duplicate a `DeviceContext`

    Not Collective

    Notes
    -----
    The duplicated `DeviceContext` shares the same options (and `Device`) but is a separate object
    and stream

    See Also
    --------
    create, destroy, petsc.PetscDeviceContextDuplicate

    """
    cdef PetscDeviceContext octx = NULL

    CHKERR(PetscDeviceContextDuplicate(self.dctx, &octx))
    return PyPetscDeviceContext_New(octx)

  def idle(self) -> bool:
    """Return whether the underlying stream for the `DeviceContext` is idle

    Not Collective

    See Also
    --------
    synchronize, petsc.PetscDeviceContextQueryIdle

    """
    cdef PetscBool is_idle = PETSC_FALSE

    CHKERR(PetscDeviceContextQueryIdle(self.dctx, &is_idle))
    return toBool(is_idle)

  def waitFor(self, other: DeviceContext | None) -> None:
    """Make this `DeviceContext` wait for `other`

    Not Collective

    Parameters
    ----------
    other
        The other `DeviceContext` to wait for

    See Also
    --------
    fork, join, petsc.PetscDeviceContextWaitForContext

    """
    cdef PetscDeviceContext cother = NULL

    if other is not None:
      cother = PyPetscDeviceContext_Get(other)
    CHKERR(PetscDeviceContextWaitForContext(self.dctx, cother))

  def fork(self, n: int, stream_type: DeviceContext.StreamType | str | None = None) -> list[DeviceContext]:
    """Create `n` `DeviceContext`s which are all logically dependent on this one

    Not Collective

    Parameters
    ----------
    n
        The number of `DeviceContext`s to create
    stream_type
        The `DeviceContext.StreamType` of the forked `DeviceContext`s

    See Also
    --------
    join, waitFor, petsc.PetscDeviceContextFork

    """
    cdef PetscDeviceContext *subctx       = NULL
    cdef PetscStreamType     cstream_type = PETSC_STREAM_DEFAULT_BLOCKING
    cdef PetscInt cn = asInt(n)
    try:
      if stream_type is None:
        CHKERR(PetscDeviceContextFork(self.dctx, cn, &subctx))
      else:
        cstream_type = asStreamType(stream_type)
        CHKERR(PetscDeviceContextForkWithStreamType(self.dctx, cstream_type, cn, &subctx))
        return [PyPetscDeviceContext_New(subctx[i]) for i in range(cn)]
    finally:
      CHKERR(PetscFree(subctx))

  def join(self, join_mode: DeviceContext.JoinMode | str, py_sub_ctxs: list[DeviceContext]) -> None:
    """Join a set of `DeviceContext`s on this one

    Not Collective

    Parameters
    ----------
    join_mode
        The type of join to perform
    py_sub_ctxs
        The list of `DeviceContext`s to join

    See Also
    --------
    fork, waitFor, petsc.PetscDeviceContextJoin

    """
    cdef PetscDeviceContext         *np_subctx_copy = NULL
    cdef PetscDeviceContext         *np_subctx      = NULL
    cdef PetscInt                    nsub           = 0
    cdef PetscDeviceContextJoinMode  cjoin_mode     = asJoinMode(join_mode)

    tmp = oarray_p(py_sub_ctxs, &nsub, <void**>&np_subctx)
    try:
      CHKERR(PetscMalloc(<size_t>(nsub) * sizeof(PetscDeviceContext *), &np_subctx_copy))
      CHKERR(PetscMemcpy(np_subctx_copy, np_subctx, <size_t>(nsub) * sizeof(PetscDeviceContext *)))
      CHKERR(PetscDeviceContextJoin(self.dctx, nsub, cjoin_mode, &np_subctx_copy))
    finally:
      CHKERR(PetscFree(np_subctx_copy))

    if cjoin_mode == PETSC_DEVICE_CONTEXT_JOIN_DESTROY:
      for i in range(nsub):
        py_sub_ctxs[i] = None

  def synchronize(self) -> None:
    """Synchronize a `DeviceContext`

    Not Collective

    Notes
    -----
    The underlying stream is considered idle after this routine returns, i.e. `idle` will return `True`

    See Also
    --------
    idle, petsc.PetscDeviceContextSynchronize

    """
    CHKERR(PetscDeviceContextSynchronize(self.dctx))

  def setFromOptions(self, comm: Comm | None = None) -> None:
    """Configure the `DeviceContext` from the options database

    Collective

    Parameters
    ----------
    comm
        The `Comm` to use (or `None` for `PETSC_COMM_SELF`)

    See Also
    --------
    petsc.PetscDeviceContextSetFromOptions

    """
    cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)

    CHKERR(PetscDeviceContextSetFromOptions(ccomm, self.dctx))

  @staticmethod
  def getCurrent() -> DeviceContext:
    """Return the current `DeviceContext`

    Not Collective

    Notes
    -----
    Reading the ``current`` instance property has the same effect as calling this routine

    See Also
    --------
    current, setCurrent, petsc.PetscDeviceContextGetCurrentContext

    """
    cdef PetscDeviceContext dctx = NULL

    CHKERR(PetscDeviceContextGetCurrentContext(&dctx))
    return PyPetscDeviceContext_New(dctx)

  @staticmethod
  def setCurrent(dctx: DeviceContext | None) -> None:
    """Set the current `DeviceContext`

    Not Collective

    Parameters
    ----------
    dctx
        The `DeviceContext` to set as current (or `None` to use the default `NULL` context)

    Notes
    -----
    Writing the ``current`` instance property has the same effect as calling this routine

    See Also
    --------
    current, getCurrent, petsc.PetscDeviceContextSetCurrentContext

    """
    cdef PetscDeviceContext cdctx = NULL

    if dctx is not None:
      cdctx = PyPetscDeviceContext_Get(dctx)
    CHKERR(PetscDeviceContextSetCurrentContext(cdctx))

  property current:
  """The current global `DeviceContxt`"""
    def __get__(self) -> DeviceContext:
      return self.getCurrent()

    def __set__(self, dctx: DeviceContext | None) -> None:
      self.setCurrent(dctx)

# --------------------------------------------------------------------

del DeviceType
del DeviceJoinMode
del StreamType
del staticproperty
