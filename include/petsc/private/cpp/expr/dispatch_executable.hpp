#ifndef PETSC_CPP_EXPR_DISPATCH_EXECUTABLE_HPP
#define PETSC_CPP_EXPR_DISPATCH_EXECUTABLE_HPP

#include <petscdevice_cupm.h>
#include <petscmanagedmemory_fwd.hpp>

#if PetscDefined(HAVE_CUPM)
  #include <petsc/private/cupminterface.hpp>
  #include "../src/sys/objects/device/impls/cupm/kernels.hpp"
#endif

#include <petsc/private/cpp/expr/expression.hpp>

#include <petsc/private/cpp/type_traits.hpp> // util::enable_if_t
#include <petsc/private/cpp/tuple.hpp>       // tuple_for_each()
#include <petsc/private/cpp/memory.hpp>      // std::addressof
#include <petsc/private/cpp/array.hpp>       // util::make_array

#include <vector>    // std::vector
#include <algorithm> // std::sort

namespace Petsc
{

namespace expr
{

namespace detail
{

template <typename T>
struct Collector {
  using vec_type = std::vector<const ManagedMemory<T> *>;

  void operator()(const ManagedMemory<T> &expr) noexcept { this->list->emplace_back(std::addressof(expr)); }

  template <typename... U>
  void operator()(const Expression<U...> &expr) noexcept
  {
    util::tuple_for_each(expr.expr(), *this);
  }

  template <typename U>
  util::enable_if_t<!std::is_same<T, U>::value> operator()(const ManagedMemory<U> &expr) noexcept
  {
    this->list->emplace_back((const ManagedMemory<T> *)std::addressof(expr));
  }

  PETSC_CONSTEXPR_14 void operator()(...) const noexcept { }

  PETSC_NODISCARD static Collector MakeCollector() noexcept
  {
    static vec_type slist{};

    slist.clear();
    return {&slist};
  }

  PETSC_NODISCARD vec_type *get_unique() noexcept
  {
    PetscCallCXXAbort(PETSC_COMM_SELF, {
      std::sort(this->list->begin(), this->list->end());
      this->list->erase(std::unique(this->list->begin(), this->list->end()), this->list->end());
    });
    return this->list;
  }

  vec_type *list;
};

template <typename T>
constexpr inline Collector<T> &MultiEvalCollect(Collector<T> &collecter) noexcept
{
  return collecter;
}

template <typename T, typename Top, typename... Rest>
PETSC_NODISCARD inline Collector<T> &MultiEvalCollect(Collector<T> &collector, Top &&top, Rest &&...rest) noexcept
{
  return MultiEvalCollect(util::tuple_for_each(top.second.expr(), collector), std::forward<Rest>(rest)...);
}

template <typename T>
PETSC_KERNEL_DECL PETSC_LAUNCH_BOUNDS(1) void ExecuteLambda(std::size_t N, T lambda)
{
#if PetscDefined(USING_CUPMCC)
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(N, std::move(lambda));
#else
  static_cast<void>(N);
  static_cast<void>(lambda);
#endif
  return;
}

template <typename CUPMInterface, typename size_type, typename T>
inline PetscErrorCode CUPMExec(PetscDeviceContext dctx, size_type N, T &&lambda) noexcept
{
  typename CUPMInterface::cupmStream_t *stream = nullptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, reinterpret_cast<void **>(&stream)));
#if PetscDefined(USING_NVCC)
  static_assert(__nv_is_extended_device_lambda_closure_type(T), "");
#endif
  PetscCall(CUPMInterface::PetscCUPMLaunchKernel1D(1, 0, *stream, ExecuteLambda<T>, N, std::forward<T>(lambda)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ExecuteLambda(PetscDeviceContext dctx, PetscDeviceType dtype, std::size_t N, T &&lambda) noexcept
{
  PetscFunctionBegin;
  static_cast<void>(dctx);
  switch (dtype) {
  case PETSC_DEVICE_HOST:
    for (std::size_t i = 0; i < N; ++i) lambda(i);
    break;
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(CUPMExec<Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::CUDA>>(dctx, N, std::forward<T>(lambda)));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(CUPMExec<Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::HIP>>(dctx, N, std::forward<T>(lambda)));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Trying to execute an expression with device type %s, but PETSc has not been configured with %s support!", PetscDeviceTypes[dtype], PetscDeviceTypes[dtype]);
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Dispatch - Actually
*/
template <typename Expr, typename T>
inline PetscErrorCode ExecutePrepare(PetscDeviceContext dctx, PetscDeviceType dtype, PetscMemType mtype, Expr &&expr, ManagedMemory<T> &dest)
{
  using value_type      = typename ManagedMemory<T>::value_type;
  auto        collector = Collector<T>::MakeCollector();
  const auto  man_vec   = util::tuple_for_each(expr.expr(), collector).get_unique();
  const auto  dest_ptr  = std::addressof(dest);
  auto        dest_mode = PETSC_MEMORY_ACCESS_WRITE;
  value_type *array     = nullptr;

  PetscFunctionBegin;
  for (auto &&man : *man_vec) {
    const value_type *dummy;

    if (man == dest_ptr) {
      dest_mode = PETSC_MEMORY_ACCESS_READ_WRITE;
    } else {
      PetscCall(man->GetArrayRead(dctx, mtype, PETSC_FALSE, &dummy));
    }
  }
  PetscCall(dest.GetArray(dctx, mtype, dest_mode, PETSC_FALSE, &array));
  {
    auto ptr_expr = expr.Freeze(mtype);

    PetscCall(ExecuteLambda(dctx, dtype, dest.size(), [array, ptr_expr] PETSC_HOSTDEVICE_DECL(std::size_t i) { array[i] = ptr_expr[i]; }));
  }
  for (auto &&man : *man_vec) {
    const value_type *dummy;

    if (man != dest_ptr) PetscCall(man->RestoreArrayRead(dctx, mtype, PETSC_FALSE, &dummy));
  }
  PetscCall(dest.RestoreArray(dctx, mtype, dest_mode, PETSC_FALSE, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ExecutePrepare(PetscDeviceContext dctx, PetscDeviceType dtype, PetscMemType mtype, const Expression<operators::identity, const ManagedMemory<T> &> &expr, ManagedMemory<T> &dest)
{
  using ptr_type       = typename ManagedMemory<T>::value_type;
  const auto      size = dest.size();
  auto          &&src  = std::get<0>(expr.expr());
  const ptr_type *src_ptr;
  ptr_type       *dest_ptr;

  PetscFunctionBegin;
  if (std::addressof(src) == std::addressof(dest)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(src.GetArrayRead(dctx, mtype, PETSC_FALSE, &src_ptr));
  PetscCall(dest.GetArray(dctx, mtype, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &dest_ptr));
  if (dtype == PETSC_DEVICE_HOST) {
    PetscCall(PetscArraycpy(dest_ptr, src_ptr, size));
  } else {
    void *stream = nullptr;

    PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
    switch (dtype) {
#if PetscDefined(HAVE_CUDA)
    case PETSC_DEVICE_CUDA:
      using CUPMInterface = Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::CUDA>;

      PetscCall(CUPMInterface::PetscCUPMMemcpyAsync(dest_ptr, src_ptr, size, CUPMInterface::cupmMemcpyDeviceToDevice, *static_cast<typename CUPMInterface::cupmStream_t *>(stream), true));
      break;
#endif
#if PetscDefined(HAVE_HIP)
    case PETSC_DEVICE_HIP:
      using CUPMInterface = Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::HIP>;

      PetscCall(CUPMInterface::PetscCUPMMemcpyAsync(dest_ptr, src_ptr, size, CUPMInterface::cupmMemcpyDeviceToDevice, *static_cast<typename CUPMInterface::cupmStream_t *>(stream), true));
      break;
#endif
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unhandled device type %s", PetscDeviceTypes[dtype]);
      break;
    }
  }
  PetscCall(src.RestoreArrayRead(dctx, mtype, PETSC_FALSE, &src_ptr));
  PetscCall(dest.RestoreArray(dctx, mtype, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &dest_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T, typename Expr, typename... ManagedMemoryExprPair>
inline PetscErrorCode MultiEvalExecutePrepare(PetscDeviceContext dctx, PetscDeviceType dtype, PetscMemType mtype, std::pair<ManagedMemory<T> &, Expr> expr1, ManagedMemoryExprPair &&...expr_n) noexcept
{
  const auto out_man_ptrs = util::make_array(std::addressof(expr1.first), std::addressof(expr_n.first)...);
  auto       collector    = Collector<T>::MakeCollector();
  const auto man_vec      = MultiEvalCollect(collector, expr1, expr_n...).get_unique();

  std::array<PetscMemoryAccessMode, out_man_ptrs.size()> out_modes;
  std::array<T *, out_man_ptrs.size()>                   out_ptrs;

  PetscFunctionBegin;
  out_modes.fill(PETSC_MEMORY_ACCESS_WRITE);
  // Need to remove any instances of the output pointers from the list of input pointers. This
  // saves us calling GetArray() (and by extension marking them) twice.
  // clang-format off
  PetscCallCXX(
    const auto out_begin = out_man_ptrs.cbegin();
    const auto out_end   = out_man_ptrs.cend();

    man_vec->erase(
      std::remove_if(
        man_vec->begin(), man_vec->end(),
        [&] (const ManagedMemory<T> *ptr) {
          const auto out_it = std::find(out_begin, out_end, ptr);

          if (out_it == out_end) {
            // pointer found not in the output, is a pure read-only operation for it
            const T *array;

            PetscCallAbort(PETSC_COMM_SELF, ptr->GetArrayRead(dctx, mtype, PETSC_FALSE, &array));
            // do not remove ptr from man_vec
            return false;
          }
          // pointer found in both the input and output, read-write
          out_modes[std::distance(out_begin, out_it)] = PETSC_MEMORY_ACCESS_READ_WRITE;
          // remove ptr from man_vec
          return true;
        }
      ),
      man_vec->end()
    )
  );
  // clang-format on

  for (std::size_t i = 0; i < out_man_ptrs.size(); ++i) PetscCall(out_man_ptrs[i]->GetArray(dctx, mtype, out_modes[i], PETSC_FALSE, &out_ptrs[i]));

  {
    auto ptr_exprs = std::make_tuple(expr1.second.Freeze(mtype), expr_n.second.Freeze(mtype)...);

    PetscCall(expr::detail::ExecuteLambda(dctx, dtype, expr1.first.size(), [out_ptrs, ptr_exprs] PETSC_HOSTDEVICE_DECL(std::size_t i) {
      std::size_t idx = 0;

      util::tuple_for_each(ptr_exprs, [&](auto &&expr) { out_ptrs[idx++][i] = expr[i]; });
    }));
  }

  for (auto &&ptr : *man_vec) {
    const T *array;

    PetscCall(ptr->RestoreArrayRead(dctx, mtype, PETSC_FALSE, &array));
  }
  for (std::size_t i = 0; i < out_man_ptrs.size(); ++i) PetscCall(out_man_ptrs[i]->RestoreArray(dctx, mtype, out_modes[i], PETSC_FALSE, &out_ptrs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace detail

} // namespace expr

} // namespace Petsc

#endif // PETSC_CPP_EXPR_DISPATCH_EXECUTABLE_HPP
