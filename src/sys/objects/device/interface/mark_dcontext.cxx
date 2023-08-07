#include "petscdevice_interface_internal.hpp" /*I <petscdevice.h> I*/

#include <petsc/private/cpp/object_pool.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/unordered_map.hpp>
#include <petsc/private/cpp/memory.hpp>

#include <algorithm> // std::remove_if(), std::find_if()
#include <vector>
#include <string>
#include <sstream> // std::ostringstream

#if defined(__clang__)
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wgnu-zero-variadic-macro-arguments")
#endif

#if PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)
  #define PETSC_HAVE_STACK 1
#endif

namespace
{

// ASYNC TODO: remove
#define TIME_EVENTS PetscDefined(USE_DEBUG)
#if PetscHasBuiltin(__builtin_ia32_rdtsc)
  #include <cstddef>
  #define PETSC_USE_IA32 1
#else
  #include <chrono>
  #define PETSC_USE_IA32 0
#endif

struct EventCounter : Petsc::RegisterFinalizeable<EventCounter> {
  std::string name;
  std::size_t cnt = 0;
#if PETSC_USE_IA32
  std::uint64_t t1;
  std::uint64_t duration = 0;
  std::uint64_t min      = 1e6;
#else
  using clock_t = std::chrono::steady_clock;
  typename clock_t::time_point t1;
  std::chrono::nanoseconds     duration{0};
  std::chrono::nanoseconds     min{std::chrono::nanoseconds::max()};
#endif

  EventCounter(std::string name) : name{std::move(name)} { }

  void tick()
  {
#if TIME_EVENTS
    (void)(this->register_finalize());
    ++cnt;
#endif
  }

  void begin()
  {
    this->tick();
#if TIME_EVENTS
  #if PETSC_USE_IA32
    t1 = __builtin_ia32_rdtsc();
  #else
    t1      = clock_t::now();
  #endif
#endif
    return;
  }

  void end()
  {
#if TIME_EVENTS
  #if PETSC_USE_IA32
    auto t2 = __builtin_ia32_rdtsc();
  #else
    auto t2 = clock_t::now();
  #endif
    auto dur = t2 - t1;
    duration += dur;
    if (dur < min) min = dur;
#endif
    return;
  }

  PetscErrorCode finalize_() noexcept
  {
#if PETSC_USE_IA32
    auto count    = (double)(duration / 2.85);
    auto mincount = (double)(min / 2.85);
#else
    auto count    = (double)duration.count();
    auto mincount = (double)min.count();
#endif
    auto        avg_count = count / cnt;
    const char *duration_name, *avg_duration_name;
    const auto  scale_count = [](double &count, const char *&dur_name) {
      if (count > 1000000000.) {
        count /= 1000000000.;
        dur_name = "s";
      } else if (count > 1000000.) {
        count /= 1000000.;
        dur_name = "ms";
      } else if (count > 1000.) {
        count /= 1000.;
        dur_name = "us";
      } else {
        dur_name = "ns";
      }
    };

    PetscFunctionBegin;
    scale_count(count, duration_name);
    scale_count(avg_count, avg_duration_name);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: %zu calls, total %g %s (min %g %s, avg %g %s)\n", name.c_str(), cnt, count, duration_name, mincount, avg_duration_name, avg_count, avg_duration_name));
    cnt = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

} // namespace

// ASYNC TODO: remove
static EventCounter wait_counter("Event wait");
// ASYNC TODO: remove
static EventCounter mark_counter("MarkID");
// ASYNC TODO: remove
static EventCounter create_counter("Event create");

namespace
{

PetscErrorCode PetscEventClearDctxEvent_Private(PetscEvent event) noexcept
{
  PetscFunctionBegin;
  if (const auto dctx = event->weak->weak_dctx().lock()) {
    auto &dctx_event = dctx->event;

    if (dctx_event == event) {
      PetscAssert(dctx_event->dctx_id == PetscObjectCast(dctx.get())->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ids don't match");
      dctx_event = nullptr;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// PetscEvent
// ==========================================================================================

class PetscEventConstructor : public Petsc::ConstructorInterface<_n_PetscEvent, PetscEventConstructor> {
  template <bool check>
  static PetscErrorCode full_reset_(PetscEvent event, PetscDeviceType dtype) noexcept
  {
    PetscFunctionBegin;
    if (check) {
      PetscAssert(event->refcnt == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Event reference count %" PetscInt64_FMT " != 0", event->refcnt);
      if (const auto destroy = event->destroy) PetscCall((*destroy)(event));
      PetscAssert(!event->data, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Event failed to destroy its data member: %p", event->data);
    }
    event->dtype   = dtype;
    event->dctx_id = 0;
    event->refcnt  = 1;
    event->pending = PETSC_FALSE;
    event->data    = nullptr;
    event->destroy = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  static PetscErrorCode construct_(PetscEvent event, PetscDeviceType dtype) noexcept
  {
    PetscFunctionBegin;
    create_counter.begin();
    PetscCall(full_reset_<false>(event, dtype));
    PetscCallCXX(event->weak = new _n_WeakContext{});
    create_counter.end();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode destroy_(PetscEvent event) noexcept
  {
    PetscFunctionBegin;
    PetscCall(full_reset_<true>(event, PETSC_DEVICE_DEFAULT()));
    delete event->weak;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode reset_(PetscEvent event, PetscDeviceType dtype) noexcept
  {
    PetscFunctionBegin;
    if (event->dtype == dtype) {
      PetscAssert(event->refcnt == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Event reference count %" PetscInt64_FMT " != 0", event->refcnt);
      event->refcnt = 1;
    } else {
      PetscCall(full_reset_<true>(event, dtype));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode invalidate_(PetscEvent event) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscEventClearDctxEvent_Private(event));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

Petsc::ObjectPool<_n_PetscEvent, PetscEventConstructor> event_pool;

} // namespace

PetscErrorCode PetscDeviceContextCreateEvent_Internal(PetscDeviceContext dctx, PetscEvent *event)
{
  PetscObject obj;
  PetscEvent  dctx_event;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscAssertPointer(event, 2);
  obj        = PetscObjectCast(dctx);
  dctx_event = dctx->event;
  if (dctx_event && (dctx_event->weak->state() >= obj->state)) {
    PetscCheck(dctx_event->dctx_id == obj->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Last recorded event was re-recorded by another device context! Event had dctx id %" PetscInt64_FMT ", expected %" PetscInt64_FMT, dctx_event->dctx_id, obj->id);
    ++dctx_event->refcnt;
    *event = dctx_event;
  } else {
    PetscCall(event_pool.allocate(event, dctx->device->type));
  }
  if (!(*event)->destroy) PetscTryTypeMethod(dctx, createevent, *event);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscEventDestroy_Internal(PetscEvent *event)
{
  PetscFunctionBegin;
  PetscAssertPointer(event, 1);
  if (!*event) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck((*event)->refcnt > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Refcount %" PetscInt64_FMT " <= 0", (*event)->refcnt);
  if (--(*event)->refcnt) {
    *event = nullptr;
  } else {
    PetscCall(event_pool.deallocate(event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextRecordEvent_Internal(PetscDeviceContext dctx, PetscEvent *ioevent)
{
  PetscEvent       old_event, event;
  PetscObjectId    id;
  PetscObjectState state;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscAssertPointer(ioevent, 2);
  PetscAssertPointer(*ioevent, 2);
  id        = PetscObjectCast(dctx)->id;
  state     = PetscObjectCast(dctx)->state;
  event     = *ioevent;
  old_event = dctx->event;
  // Note: we CANNOT just compare event->dctx and dctx here. dctx may be recycled, and would
  // therefore be a logically distinct object. So we must check the saved id and state.
  if (old_event == event) {
    // do nothing, we just recorded this event (we will likely bail below)
  } else if (old_event && (old_event->dtype == event->dtype) && (state == old_event->weak->state())) {
    // we can replace the event with one we just recorded to
    ++old_event->refcnt;
    PetscCall(PetscEventDestroy_Internal(ioevent));
    *ioevent = event = old_event;
  }
  if (id == event->dctx_id && state == event->weak->state()) PetscFunctionReturn(PETSC_SUCCESS);
  // REVIEW ME:
  // TODO maybe move this to impls, as they can determine whether they can interoperate with
  // other device types more readily
  if (PetscDefined(USE_DEBUG) && (event->dtype != PETSC_DEVICE_HOST)) {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCheck(event->dtype == dtype, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Event type %s does not match device context type %s", PetscDeviceTypes[event->dtype], PetscDeviceTypes[dtype]);
  }
  if (event->dctx_id != id) {
    PetscCall(PetscEventClearDctxEvent_Private(event));
    event->dctx_id = id;
  }
  event->pending = PETSC_TRUE;
  *event->weak   = CxxDataCast(dctx)->weak_snapshot();
  dctx->event    = event;
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace
{

PetscErrorCode PetscDeviceContextWaitForEventUnequalID_Private(PetscDeviceContext dctx, PetscEvent event) noexcept
{
  const auto event_id  = event->dctx_id;
  const auto weak_data = event->weak;
  auto     &&upstream  = CxxDataCast(dctx)->upstream();
  const auto it        = upstream.find(event_id);

  PetscFunctionBegin;
  if (it == upstream.end()) {
    PetscCallCXX(upstream[event_id] = *weak_data);
  } else {
    const auto weak_state         = weak_data->state();
    auto      &upstream_weak_data = it->second;

    if (upstream_weak_data.state() >= weak_state) {
      // we have either waited on this exact event before, or we have previously waited on
      // "newer" events recorded by event->dctx
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    upstream_weak_data.set_state(weak_state);
  }

  if (event->pending) {
    const auto event_dctx = weak_data->weak_dctx().lock().get();

    PetscCheck(event_dctx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dctx %" PetscInt64_FMT " was destroyed before recording pending event", event_id);
    PetscCall(PetscDeviceContextForceEventRecord_Internal(event_dctx));
    PetscCheck(event->pending == PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dctx %" PetscInt64_FMT " failed to record pending event!", PetscObjectCast(event_dctx)->id);
  }
  wait_counter.begin();
  PetscTryTypeMethod(dctx, waitforevent, event);
  wait_counter.end();
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode PetscDeviceContextWaitForEvent_Internal(PetscDeviceContext dctx, PetscEvent event)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  if (!event) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(event, 2);
  // empty data implies you cannot wait on this event
  if (!event->data) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscDefined(USE_DEBUG)) {
    const auto      etype = event->dtype;
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCheck(etype == dtype, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Event type %s does not match device context type %s", PetscDeviceTypes[etype], PetscDeviceTypes[dtype]);
  }
  if (PetscObjectCast(dctx)->id != event->dctx_id) PetscCall(PetscDeviceContextWaitForEventUnequalID_Private(dctx, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace
{

// ==========================================================================================
// PetscStackFrame
//
// A helper class that (when debugging is enabled) contains the stack frame from which
// PetscDeviceContextMarkIntentFromID(). It is intended to be derived from, since this enables
// empty-base-class optimization to kick in when debugging is disabled.
// ==========================================================================================

template <bool have_stack>
struct PetscStackFrame;

template <>
struct PetscStackFrame</* have_stack = */ true> {
  std::string file{};
  std::string function{};
  int         line{};

  PetscStackFrame() = default;

  PetscStackFrame(const char *file_, const char *func_, int line_) noexcept : file{split_on_petsc_path_(file_)}, function{func_}, line{line_} { }

  bool operator==(const PetscStackFrame &other) const noexcept { return line == other.line && file == other.file && function == other.function; }

  PETSC_NODISCARD std::string to_string() const noexcept
  {
    std::string ret;

    ret = '(' + function + "() at " + file + ':' + std::to_string(line) + ')';
    return ret;
  }

  void swap(PetscStackFrame &other) noexcept
  {
    using std::swap;

    file.swap(other.file);
    function.swap(other.function);
    swap(line, other.line);
  }

  friend void swap(PetscStackFrame &lhs, PetscStackFrame &rhs) noexcept PETSC_UNUSED { lhs.swap(rhs); }

private:
  static std::string split_on_petsc_path_(std::string &&in) noexcept
  {
    auto pos = in.find("petsc/src");

    if (pos == std::string::npos) pos = in.find("petsc/include");
    if (pos == std::string::npos) pos = 0;
    return in.substr(pos);
  }

  friend std::ostream &operator<<(std::ostream &os, const PetscStackFrame &frame) PETSC_UNUSED
  {
    os << frame.to_string();
    return os;
  }
};

template <>
struct PetscStackFrame</* have_stack = */ false> {
  template <typename... T>
  constexpr PetscStackFrame(T &&...) noexcept
  {
  }

  constexpr bool operator==(const PetscStackFrame &) const noexcept { return true; }

  PETSC_NODISCARD static const std::string &to_string() noexcept
  {
    static const std::string ret = "(unknown)";
    return ret;
  }

  friend std::ostream &operator<<(std::ostream &os, const PetscStackFrame &frame) noexcept PETSC_UNUSED
  {
    os << frame.to_string();
    return os;
  }

  PETSC_CONSTEXPR_14 void swap(const PetscStackFrame &) const noexcept { }
};

template <bool use_debug>
class DebugPrinter;

template <>
class DebugPrinter</* use_debug = */ true> {
public:
  DebugPrinter(PetscDeviceContext dctx, PetscObjectId id, const char name[]) : dctx_{dctx}
  {
    const auto         pobj = PetscObjectCast(dctx);
    const char        *dctx_name;
    PetscObjectId      dctx_id;
    std::ostringstream oss;
    MPI_Comm           comm;

    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, PetscObjectGetComm(PetscObjectCast(dctx_), &comm));
    PetscCallAbort(comm, PetscObjectGetName(pobj, &dctx_name));
    PetscCallAbort(comm, PetscObjectGetId(pobj, &dctx_id));
    PetscCallCXXAbort(comm, oss << "dctx " << dctx_id << " (" << dctx_name << ") - obj " << id << " (" << name << "): ");
    PetscCallCXXAbort(comm, preamble_ = oss.str());
    PetscFunctionReturnVoid();
  }

  template <typename... T>
  PetscErrorCode operator()(const char str[], T &&...args) const noexcept
  {
    PetscFunctionBegin;
    PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wformat-security");
    PetscCall(PetscDebugInfo(dctx_, (preamble_ + str).c_str(), std::forward<T>(args)...));
    PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  PetscDeviceContext dctx_{};
  std::string        preamble_{};
};

template <>
class DebugPrinter</* use_debug = */ false> {
public:
  template <typename... T>
  constexpr DebugPrinter(T &&...) noexcept
  {
  }

  template <typename... T>
  constexpr PetscErrorCode operator()(T &&...) const noexcept
  {
    return PETSC_SUCCESS;
  }
};

// ==========================================================================================
// MarkedObjectMap
//
// A mapping from a PetscObjectId to a PetscEvent and (if debugging is enabled) a
// PetscStackFrame containing the location where PetscDeviceContextMarkIntentFromID was called
// ==========================================================================================

class MarkedObjectMap : public Petsc::RegisterFinalizeable<MarkedObjectMap> {
public:
  using FrameType        = PetscStackFrame<PetscDefined(HAVE_STACK)>;
  using DebugPrinterType = DebugPrinter<PetscDefined(USE_DEBUG_AND_INFO)>;
  // Note we derive from PetscStackFrame so that the empty base class optimization can kick
  // in. If it were just a member it would still take up storage in optimized builds
  class SnapshotType : private FrameType {
    struct PetscEventDeleter {
      void operator()(PetscEvent event) const noexcept
      {
        PetscFunctionBegin;
        PetscCallAbort(PETSC_COMM_SELF, PetscEventDestroy_Internal(&event));
        PetscFunctionReturnVoid();
      }
    };

  public:
    std::unique_ptr<_n_PetscEvent, PetscEventDeleter> event{};

    SnapshotType() = default;
    SnapshotType(PetscDeviceContext, bool, FrameType) noexcept;

    PETSC_NODISCARD const FrameType &frame() const noexcept { return *this; }
    PETSC_NODISCARD FrameType       &frame() noexcept { return *this; }

    void        swap(SnapshotType &) noexcept;
    friend void swap(SnapshotType &lhs, SnapshotType &rhs) noexcept PETSC_UNUSED { lhs.swap(rhs); }
  };

  // the "value" each key maps to
  struct MappedType {
    using DependencyType = std::vector<SnapshotType>;

    MappedType() noexcept;

    // The memory access mode this object was last marked with. Note the defalt is read to
    // ensure that any async accesses are fully coherent with preceeding synchronous access
    PetscMemoryAccessMode mode{PETSC_MEMORY_ACCESS_READ};
    // The last write dependency of the object. This is not necessarily empty if mode is
    // PETSC_MEMORY_ACCESS_READ since every additional reader needs to synchronize on the last
    // writer
    SnapshotType write_dep{};
    // The set of read dependencies. Any writer must sync with all readers (and clear them) in
    // order to become write_dep
    DependencyType read_deps{};

    void        swap(MappedType &) noexcept;
    friend void swap(MappedType &lhs, MappedType &rhs) noexcept PETSC_UNUSED { lhs.swap(rhs); }
  };

  using MapType = Petsc::UnorderedMap<PetscObjectId, MappedType>;

  template <bool>
  PetscErrorCode Mark(PetscDeviceContext, PetscObjectId, PetscMemoryAccessMode, const char[]) noexcept;
  PetscErrorCode RemoveMark(PetscObjectId) noexcept;
  template <typename T>
  PetscErrorCode IterVisitor(PetscDeviceContext, T &&) noexcept;
  PetscErrorCode Export(std::size_t *, PetscObjectId **, PetscMemoryAccessMode **, PetscEvent **, std::size_t **, PetscEvent ***) const noexcept;
  PetscErrorCode ExportRestore(std::size_t, PetscObjectId **, PetscMemoryAccessMode **, PetscEvent **, std::size_t **, PetscEvent ***) const noexcept;

private:
  friend RegisterFinalizeable;

  PetscErrorCode finalize_() noexcept;

  template <bool>
  static PetscErrorCode UpdateOrReplaceSnapshot_(SnapshotType &, PetscDeviceContext, FrameType &, const DebugPrinterType &) noexcept;
  // The current mode is compatible with the previous mode (i.e. read-read) so we need only
  // update the existing version and possibly appeand ourselves to the dependency list
  template <bool>
  static PetscErrorCode MarkCompatibleModes_(MappedType &, PetscDeviceContext, PetscMemoryAccessMode, FrameType &, const DebugPrinterType &, bool *) noexcept;
  // The current mode is NOT compatible with the previous mode. We must serialize with all events
  // in the dependency list, possibly clear it, and update the previous write event
  template <bool>
  static PetscErrorCode MarkIncompatibleModes_(MappedType &, PetscDeviceContext, PetscMemoryAccessMode, FrameType &, const DebugPrinterType &, bool *, bool *) noexcept;
  template <bool>
  PetscErrorCode Mark_(PetscDeviceContext, PetscObjectId, PetscMemoryAccessMode, FrameType &&, const char[]) noexcept;

  MapType map_;
};

// ==========================================================================================
// MarkedObjectMap::SnapshotType -- Public API
// ==========================================================================================

MarkedObjectMap::SnapshotType::SnapshotType(PetscDeviceContext dctx, bool record, FrameType frame) noexcept : FrameType{std::move(frame)}
{
  PetscFunctionBegin;
  if (record) {
    PetscEvent tmp;

    PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextCreateEvent_Internal(dctx, &tmp));
    PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextRecordEvent_Internal(dctx, &tmp));
    PetscCallCXXAbort(PETSC_COMM_SELF, this->event.reset(tmp));
  }
  PetscFunctionReturnVoid();
}

void MarkedObjectMap::SnapshotType::swap(SnapshotType &other) noexcept
{
  PetscFunctionBegin;
  PetscCallCXXAbort(PETSC_COMM_SELF, this->frame().swap(other.frame()));
  PetscCallCXXAbort(PETSC_COMM_SELF, this->event.swap(other.event));
  PetscFunctionReturnVoid();
}

// ==========================================================================================
// MarkedObjectMap::MappedType -- Public API
// ==========================================================================================

// workaround for clang bug that produces the following warning
//
// src/sys/objects/device/interface/mark_dcontext.cxx:253:5: error: default member initializer
// for 'mode' needed within definition of enclosing class 'MarkedObjectMap' outside of member
// functions
//     MappedType() noexcept = default;
//     ^
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
MarkedObjectMap::MappedType::MappedType() noexcept = default;

void MarkedObjectMap::MappedType::swap(MappedType &other) noexcept
{
  using std::swap;

  PetscFunctionBegin;
  PetscCallCXXAbort(PETSC_COMM_SELF, swap(mode, other.mode));
  PetscCallCXXAbort(PETSC_COMM_SELF, write_dep.swap(other.write_dep));
  PetscCallCXXAbort(PETSC_COMM_SELF, read_deps.swap(other.read_deps));
  PetscFunctionReturnVoid();
}

// ==========================================================================================
// MarkedObjectMap -- Private API
// ==========================================================================================

PetscErrorCode MarkedObjectMap::finalize_() noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(nullptr, "Finalizing marked object map\n"));
  // ASYNC TODO: remove
  if (PetscDefined(USE_DEBUG)) {
    std::unordered_map<void *, std::vector<PetscObjectId>> events;
    const auto                                             add_event = [&](PetscObjectId id, PetscEvent e) {
      if (e) events[e].push_back(id);
    };

    events.reserve(map_.size());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Marked Object Map destruction: %zu objects still marked\n", map_.size()));
    for (auto it = map_.begin(); it != map_.end(); ++it) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "- %" PetscInt64_FMT " %s\n", it->first, it->second.write_dep.frame().to_string().c_str()));
      add_event(it->first, it->second.write_dep.event.get());
      for (auto &&d : it->second.read_deps) add_event(it->first, d.event.get());
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%zu events:\n", events.size()));
    for (auto &&ev : events) {
      const auto &event = ev.second;

      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "- %p {", (void *)ev.first));
      for (std::size_t i = 0; i < event.size(); ++i) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt64_FMT, event[i]));
        if (i + 1 < event.size()) PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", "));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "}\n"));
    }
  }
  PetscCall(map_.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool record>
PetscErrorCode MarkedObjectMap::UpdateOrReplaceSnapshot_(SnapshotType &snapshot, PetscDeviceContext dctx, FrameType &frame, const DebugPrinterType &DEBUG_INFO) noexcept
{
  auto &event = snapshot.event;

  PetscFunctionBegin;
  if (event && (event->dtype == dctx->device->type)) {
    // Match the device type, can reuse the event! All we must do is update the frame, and
    // re-record the event
    PetscCall(DEBUG_INFO("updating snapshot\n"));
    PetscCallCXX(snapshot.frame().swap(frame));
    if (record) {
      auto event_ptr = event.release();

      PetscCall(PetscDeviceContextRecordEvent_Internal(dctx, &event_ptr));
      PetscCallCXX(event.reset(event_ptr));
    }
  } else {
    PetscCall(DEBUG_INFO("creating snapshot\n"));
    PetscCallCXX(snapshot = SnapshotType{dctx, record, std::move(frame)});
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool record>
PetscErrorCode MarkedObjectMap::MarkCompatibleModes_(MappedType &marked, PetscDeviceContext dctx, PetscMemoryAccessMode mode, FrameType &frame, const DebugPrinterType &DEBUG_INFO, bool *update_rdeps) noexcept
{
  const auto dctx_id = PetscObjectCast(dctx)->id;
  auto      &rdeps   = marked.read_deps;
  const auto end     = rdeps.end();
  // clang-format off
  const auto it = std::find_if(
    rdeps.begin(), end,
    [dctx_id](const SnapshotType &obj) { return obj.event && (obj.event->dctx_id == dctx_id); }
  );
  // clang-format on

  PetscFunctionBegin;
  PetscCall(DEBUG_INFO("new mode (%s) COMPATIBLE with old mode (%s), no need to serialize\n", PetscMemoryAccessModeToString(mode), PetscMemoryAccessModeToString(marked.mode)));
  if (it == end) {
    // we have not been here before, need to serialize with the last write event (if it exists)
    // and add ourselves to the dependency list
    PetscCall(PetscDeviceContextWaitForEvent_Internal(dctx, marked.write_dep.event.get()));
  } else {
    *update_rdeps = false;
    // we have been here before, all we must do is update our entry then we can bail
    PetscCall(DEBUG_INFO("found old self as dependency, updating\n"));
    // will be guaranteed to update instead of replace.
    PetscCall(UpdateOrReplaceSnapshot_<record>(*it, dctx, frame, DEBUG_INFO));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool record>
PetscErrorCode MarkedObjectMap::MarkIncompatibleModes_(MappedType &marked, PetscDeviceContext dctx, PetscMemoryAccessMode mode, FrameType &frame, const DebugPrinterType &DEBUG_INFO, bool *update_rdeps, bool *reuse_rdeps) noexcept
{
  const auto  old_mode  = Petsc::util::exchange(marked.mode, mode);
  const auto &rdeps     = marked.read_deps;
  auto       &write_dep = marked.write_dep;

  PetscFunctionBegin;
  // we are NOT compatible with the previous mode
  PetscCall(DEBUG_INFO("new mode (%s) NOT COMPATIBLE with old mode (%s), serializing then clearing (%zu) dependencies\n", PetscMemoryAccessModeToString(mode), PetscMemoryAccessModeToString(old_mode), rdeps.size()));
  if (rdeps.empty()) {
    // if event is nullptr then write_dep was default-constructed, i.e. nobody has ever
    // written to this object
    PetscCall(PetscDeviceContextWaitForEvent_Internal(dctx, write_dep.event.get()));
  } else {
    // if rdeps are not empty this implies last event was some kind of read, which will already
    // have waited on write_dep's event. So we can skip doing that.
    for (const auto &dep : rdeps) PetscCall(PetscDeviceContextWaitForEvent_Internal(dctx, dep.event.get()));
    *reuse_rdeps = true;
  }

  if (PetscMemoryAccessWrite(mode)) {
    *update_rdeps = false;
    // if we write, we must become write_dep
    PetscCall(UpdateOrReplaceSnapshot_<record>(write_dep, dctx, frame, DEBUG_INFO));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool record>
PetscErrorCode MarkedObjectMap::Mark_(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, FrameType &&frame, const char name[]) noexcept
{
  const auto DEBUG_INFO   = DebugPrinterType{dctx, id, name};
  const auto cxx_data     = CxxDataCast(dctx);
  auto      &marked       = map_[id];
  auto      &rdeps        = marked.read_deps;
  auto       update_rdeps = true;
  auto       reuse_rdeps  = false;

  PetscFunctionBegin;
  if (marked.mode == PETSC_MEMORY_ACCESS_READ && mode == PETSC_MEMORY_ACCESS_READ) {
    PetscCall(MarkCompatibleModes_<record>(marked, dctx, mode, frame, DEBUG_INFO, &update_rdeps));
    if (PetscDefined(USE_DEBUG) && !update_rdeps) {
      /*
        check that either the device context has marked before, or if not, that it has a
        recorded event and that event still points to itself. If the latter case this
        indicates that the event was reused for another object record. I.e.

        // an event is created and recorded by dctx
        mem_read_write(dctx, x);
        // since the object state of dctx is unchanged it reuses the recorded event from the
        // previous call here
        mem_read(dctx, y);

        Note that if we change the above to

        mem_read_write(dctx, x);
        PetscObjectStateIncrease(dctx);
        mem_read(dctx, y);

        Then (we expect) CxxDataCast(dctx)->has_marked(id) must be true, since it should NOT
        reuse the event.
      */
      const auto oid          = PetscObjectCast(dctx)->id;
      const auto has_marked   = cxx_data->has_marked(id);
      const auto event_reused = dctx->event && (dctx->event->dctx_id == oid);

      PetscCheck(has_marked || event_reused, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext %" PetscInt64_FMT " listed as dependency for object %" PetscInt64_FMT " (%s), but does not have the object in private dependency list!", oid, id, name);
    }
  } else {
    PetscCall(MarkIncompatibleModes_<record>(marked, dctx, mode, frame, DEBUG_INFO, &update_rdeps, &reuse_rdeps));
  }

  if (update_rdeps) {
    // become the new leaf by appending ourselves
    PetscCall(DEBUG_INFO("%s with intent %s\n", rdeps.empty() ? "dependency list is empty, creating new leaf" : "appending to existing leaves", PetscMemoryAccessModeToString(mode)));
    if (reuse_rdeps) {
      PetscCallCXX(rdeps.resize(1));
      PetscCall(UpdateOrReplaceSnapshot_<record>(rdeps.front(), dctx, frame, DEBUG_INFO));
    } else if (record) {
      PetscCallCXX(rdeps.emplace_back(dctx, record, std::move(frame)));
    }
  } else if (reuse_rdeps) {
    // New mode is some kind of write, but we already recorded to marked.write_dep so clear
    // the rdeps
    PetscCallCXX(rdeps.clear());
  }
  if (update_rdeps || reuse_rdeps) PetscCall(cxx_data->add_mark(id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MarkedObjectMap -- Public API
// ==========================================================================================

template <bool record>
PetscErrorCode MarkedObjectMap::Mark(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, const char name[]) noexcept
{
#if PetscDefined(HAVE_STACK)
  const auto index    = petscstack.currentsize > 3 ? petscstack.currentsize - 3 : 0;
  const auto file     = petscstack.file[index];
  const auto function = petscstack.function[index];
  const auto line     = petscstack.line[index];
#else
  constexpr const char *file     = nullptr;
  constexpr const char *function = nullptr;
  constexpr auto        line     = 0;
#endif

  PetscFunctionBegin;
  // stack memory is always fully host-device synchronous (even for async functions) so we do
  // not need to interact with the mark system for it
  if (id == PETSC_STACK_MEMORY_ID) PetscFunctionReturn(PETSC_SUCCESS);
  mark_counter.begin();
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheck(id != PETSC_DELETED_MEMORY_ID, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to mark a deleted memory region");
  PetscCheck(id != PETSC_UNKNOWN_MEMORY_ID, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to mark an unknown memory region");
  if (name) PetscAssertPointer(name, 4);
  if (record) PetscCall(this->register_finalize());
  PetscCall(PetscLogEventBegin(DCONTEXT_Mark, dctx, nullptr, nullptr, nullptr));
  PetscCall(Mark_<record>(dctx, id, mode, FrameType{file, function, line}, name ? name : "unknown object"));
  PetscCall(PetscLogEventEnd(DCONTEXT_Mark, dctx, nullptr, nullptr, nullptr));
  mark_counter.end();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MarkedObjectMap::RemoveMark(PetscObjectId id) noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(map_.erase(id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
PetscErrorCode MarkedObjectMap::IterVisitor(PetscDeviceContext dctx, T &&callback) noexcept
{
  const auto dctx_id = PetscObjectCast(dctx)->id;
  auto     &&marked  = CxxDataCast(dctx)->marked_objects();

  PetscFunctionBegin;
  for (auto &&obj : marked) {
    const auto mapit = map_.find(obj);

    // Need this check since the final PetscDeviceContext may run through this *after* the map
    // has been finalized (and cleared), and hence might fail to find its read_deps. This is
    // perfectly valid since the user no longer cares about dangling read_deps after PETSc is
    // finalized
    if (PetscLikely(mapit != map_.end())) {
      auto &deps = mapit->second.read_deps;
      // clang-format off
      const auto it = std::remove_if(
        deps.begin(), deps.end(),
        [&](const MarkedObjectMap::SnapshotType &obj)
        {
          return obj.event && (obj.event->dctx_id == dctx_id);
        }
      );
      // clang-format off

      PetscCall(callback(mapit, deps.cbegin(), static_cast<decltype(deps.cend())>(it)));
      // remove ourselves
      PetscCallCXX(deps.erase(it, deps.end()));
      // continue to next object, but erase this one if it has no more read_deps.
      if (deps.empty()) {
        if (const auto &event = mapit->second.write_dep.event) {
          // if we are not the last writer, do not erase the entry
          if (event->dctx_id != dctx_id) continue;
        }
        // either no write_dep, or we recorded write_dep ourselves, in either case we can
        // delete the entry
        PetscCallCXX(map_.erase(mapit));
      }
    }
  }
  PetscCallCXX(marked.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MarkedObjectMap::Export(std::size_t *nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, PetscEvent **write_deps, std::size_t **ndeps, PetscEvent ***read_deps) const noexcept
{
  std::size_t i    = 0;
  const auto  size = *nkeys = map_.size();

  PetscFunctionBegin;
  PetscCall(PetscMalloc5(size, keys, size, modes, size, ndeps, size, write_deps, size, read_deps));
  for (auto &&it : map_) {
    std::size_t j = 0;

    (*keys)[i]         = it.first;
    (*modes)[i]        = it.second.mode;
    (*ndeps)[i]        = it.second.read_deps.size();
    (*write_deps)[i]  = it.second.write_dep.event.get();
    (*read_deps)[i] = nullptr;
    PetscCall(PetscMalloc1((*ndeps)[i], (*read_deps) + i));
    for (auto &&dep : it.second.read_deps) (*read_deps)[i][j++] = dep.event.get();
    ++i;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MarkedObjectMap::ExportRestore(std::size_t nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, PetscEvent **write_deps, std::size_t **ndeps, PetscEvent ***read_deps) const noexcept
{
  PetscFunctionBegin;
  for (std::size_t i = 0; i < nkeys; ++i) PetscCall(PetscFree((*read_deps)[i]));
  PetscCall(PetscFree5(*keys, *modes, *ndeps, *write_deps, *read_deps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// A mapping between PetscObjectId (i.e. some PetscObject) to the list of PetscEvent's encoding
// the last time the PetscObject was accessed
MarkedObjectMap ObjMap;

} // namespace

// ==========================================================================================
// Utility Functions
// ==========================================================================================

PetscErrorCode PetscGetMarkedObjectMap_Internal(std::size_t *nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, PetscEvent **write_deps, std::size_t **ndeps, PetscEvent ***read_deps)
{
  PetscFunctionBegin;
  PetscCall(ObjMap.Export(nkeys, keys, modes, write_deps, ndeps, read_deps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRestoreMarkedObjectMap_Internal(std::size_t nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, PetscEvent **write_deps, std::size_t **ndeps, PetscEvent ***read_deps)
{
  PetscFunctionBegin;
  PetscCall(ObjMap.ExportRestore(nkeys, keys, modes, write_deps, ndeps, read_deps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSyncClearMap_Internal(PetscDeviceContext dctx)
{
  using map_iterator = MarkedObjectMap::MapType::const_iterator;
  using dep_iterator = MarkedObjectMap::MappedType::DependencyType::const_iterator;
  auto visitor = [&](map_iterator mapit, dep_iterator it, dep_iterator end) {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      std::ostringstream oss;
      const auto         mode = PetscMemoryAccessModeToString(mapit->second.mode);

      PetscCallCXX(oss << "synced dctx " << PetscObjectCast(dctx)->id << ", remaining leaves for obj " << mapit->first << ": {");
      while (it != end) {
        if (const auto &event = it->event) {
          PetscCallCXX(oss << "[dctx " << event->dctx_id << ", " << mode << ' ' << it->frame() << ']');
        }
        if (++it != end) PetscCallCXX(oss << ", ");
      }
      PetscCallCXX(oss << '}');
      PetscCall(PetscInfo(dctx, "%s\n", oss.str().c_str()));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  PetscFunctionBegin;
  if (const auto event = dctx->event) {
    if (event->dctx_id == PetscObjectCast(dctx)->id) event->pending = PETSC_FALSE;
  }
  PetscCall(ObjMap.IterVisitor(dctx, std::move(visitor)));
  {
    // the recursive sync clear map call is unbounded in case of a dependent loop so we make a
    // copy
    const auto cxx_data = CxxDataCast(dctx);
    // clang-format off
    const std::vector<CxxData::upstream_type::value_type> upstream_copy(
      std::make_move_iterator(cxx_data->upstream().begin()),
      std::make_move_iterator(cxx_data->upstream().end())
    );
    // clang-format on

    // aftermath, clear our set of parents (to avoid infinite recursion) and mark ourselves as no
    // longer contained (while the empty graph technically *is* always contained, it is not what
    // we mean by it)
    PetscCall(cxx_data->clear());
    //dctx->contained = PETSC_FALSE;
    for (auto &&upstrm : upstream_copy) {
      if (const auto udctx = upstrm.second.weak_dctx().lock()) {
        // check that this parent still points to what we originally thought it was
        PetscCheck(upstrm.first == PetscObjectCast(udctx.get())->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Upstream dctx %" PetscInt64_FMT " no longer exists, now has id %" PetscInt64_FMT, upstrm.first, PetscObjectCast(udctx.get())->id);
        PetscCall(PetscDeviceContextSyncClearMap_Internal(udctx.get()));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextCheckNotOrphaned_Internal(PetscDeviceContext dctx)
{
  std::ostringstream oss;
  //const auto         allow = dctx->options.allow_orphans, contained = dctx->contained;
  const auto allow = true, contained = true;
  auto       wrote_to_oss = false;
  using map_iterator      = MarkedObjectMap::MapType::const_iterator;
  using dep_iterator      = MarkedObjectMap::MappedType::DependencyType::const_iterator;

  PetscFunctionBegin;
  PetscCall(ObjMap.IterVisitor(dctx, [&](map_iterator mapit, dep_iterator it, dep_iterator end) {
    PetscFunctionBegin;
    if (allow || contained) PetscFunctionReturn(PETSC_SUCCESS);
    wrote_to_oss = true;
    oss << "- PetscObject (id " << mapit->first << "), intent " << PetscMemoryAccessModeToString(mapit->second.mode) << ' ' << it->frame();
    if (std::distance(it, end) == 0) oss << " (orphaned)"; // we were the only dependency
    oss << '\n';
    PetscFunctionReturn(PETSC_SUCCESS);
  }));
  PetscCheck(!wrote_to_oss, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Destroying PetscDeviceContext ('%s', id %" PetscInt64_FMT ") would leave the following dangling (possibly orphaned) dependents:\n%s\nMust synchronize before destroying it, or allow it to be destroyed with orphans",
             PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", PetscObjectCast(dctx)->id, oss.str().c_str());
  PetscCall(CxxDataCast(dctx)->clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextMarkIntentFromIDBegin - Indicate a `PetscDeviceContext`s access intent to
  the auto-dependency system

  Not Collective

  Input Parameters:
+ dctx - The `PetscDeviceContext`
. id   - The `PetscObjectId` to mark
. mode - The desired access intent
- name - The object name (for debug purposes, ignored in optimized builds)

  Notes:
  Must be followed by `PetscDeviceContextMarkIntentFromIDEnd()`!

  This routine formally informs the dependency system that `dctx` will access the object
  represented by `id` with `mode` and adds `dctx` to `id`'s list of dependencies (termed
  "leaves").

  If the existing set of leaves have an incompatible `PetscMemoryAccessMode` to `mode`, `dctx`
  will be serialized against them.

  Level: intermediate

.seealso: `PetscDeviceContextWaitForContext()`, `PetscDeviceContextSynchronize()`,
`PetscObjectGetId()`, `PetscMemoryAccessMode`
@*/
PetscErrorCode PetscDeviceContextMarkIntentFromIDBegin(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, const char name[])
{
  PetscFunctionBegin;
  PetscCall(ObjMap.Mark<false>(dctx, id, mode, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ASYNC TODO: document
PetscErrorCode PetscDeviceContextMarkIntentFromIDEnd(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, const char name[])
{
  PetscFunctionBegin;
  PetscCall(ObjMap.Mark<true>(dctx, id, mode, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace
{

template <typename T>
PetscErrorCode PetscDeviceContextMarkIntentFromIDGroup(PetscDeviceContext dctx, PetscInt n, const PetscObjectId *ids, const PetscMemoryAccessMode *modes, const char *const names[], T &&MarkFn) noexcept
{
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(ids, 3);
  PetscAssertPointer(modes, 4);
  if (names) PetscAssertPointer(names, 5);
  if (n == 1) {
    PetscCall(MarkFn(dctx, *ids, *modes, names ? *names : nullptr));
  } else {
    using tuple_type = std::tuple<PetscObjectId, PetscMemoryAccessMode, const char *>;
    std::vector<tuple_type> seen;

    for (PetscInt i = 0; i < n; ++i) {
      const auto id   = ids[i];
      const auto mode = modes[i];
      const auto name = names ? names[i] : nullptr;
      const auto end  = seen.end();
      auto       it   = std::find_if(seen.begin(), end, [=](const tuple_type &tuple) { return std::get<0>(tuple) == id; });

      if (it == end) {
        seen.emplace_back(id, mode, name);
      } else {
        *it = std::make_tuple(id, mode, name);
      }
    }
    for (auto &&tuple : seen) PetscCall(MarkFn(dctx, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode PetscDeviceContextMarkIntentFromIDBeginGroup(PetscDeviceContext dctx, PetscInt n, const PetscObjectId *ids, const PetscMemoryAccessMode *modes, const char *const names[])
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMarkIntentFromIDGroup(dctx, n, ids, modes, names, PetscDeviceContextMarkIntentFromIDBegin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextMarkIntentFromIDEndGroup(PetscDeviceContext dctx, PetscInt n, const PetscObjectId *ids, const PetscMemoryAccessMode *modes, const char *const names[])
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMarkIntentFromIDGroup(dctx, n, ids, modes, names, PetscDeviceContextMarkIntentFromIDEnd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextClearIntentFromID(PetscObjectId id)
{
  PetscFunctionBegin;
  PetscCall(ObjMap.RemoveMark(id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(__clang__)
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()
#endif
