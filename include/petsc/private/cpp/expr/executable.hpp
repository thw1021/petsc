#ifndef PETSC_CPP_EXPR_EXECUTABLE_HPP
#define PETSC_CPP_EXPR_EXECUTABLE_HPP

#include <petscmanagedmemory_fwd.hpp>

#include <petsc/private/deviceimpl.h> // PetscDeviceContextGetOptionalNullContext_Internal()

#include <petsc/private/cpp/expr/base.hpp> // is_expression
#include <petsc/private/cpp/expr/expression.hpp>
#include <petsc/private/cpp/expr/operators.hpp>
#include <petsc/private/cpp/expr/dispatch_executable.hpp>

#include <petsc/private/cpp/type_traits.hpp> // util::enable_if_t

namespace Petsc
{

namespace expr
{

// ==========================================================================================
// ExecutableExpression
// ==========================================================================================

template <typename T>
class ExecutableExpression {
public:
  static_assert(is_expression<T>::value, "");
  using expression_type = T;
  using size_type       = typename T::size_type;

  template <typename U>
  ExecutableExpression(U &&, PetscDeviceContext) noexcept;
  ExecutableExpression()                                        = delete;
  ExecutableExpression(const ExecutableExpression &)            = delete;
  ExecutableExpression(ExecutableExpression &&)                 = delete;
  ExecutableExpression &operator=(const ExecutableExpression &) = delete;
  ExecutableExpression &operator=(ExecutableExpression &&)      = delete;

  PETSC_NODISCARD const expression_type &expr() const noexcept;
  PETSC_NODISCARD PetscDeviceContext     dctx() const noexcept;
  PETSC_NODISCARD size_type              size() const noexcept;

  template <typename U>
  PetscErrorCode Execute(ManagedMemory<U> &, PetscDeviceContext) const noexcept;

  template <typename U>
  PetscErrorCode Execute(ManagedMemory<U> &) const noexcept;

private:
  expression_type    expr_;
  PetscDeviceContext dctx_;
};

// ==========================================================================================
// ExecutableExpression -- Public API
// ==========================================================================================

template <typename T>
template <typename U>
inline ExecutableExpression<T>::ExecutableExpression(U &&expr, PetscDeviceContext dctx) noexcept : expr_{std::forward<U>(expr)}, dctx_{dctx}
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextGetOptionalNullContext_Internal(&dctx_));
  PetscFunctionReturnVoid();
}

template <typename T>
inline PetscDeviceContext ExecutableExpression<T>::dctx() const noexcept
{
  return dctx_;
}

template <typename T>
inline typename ExecutableExpression<T>::size_type ExecutableExpression<T>::size() const noexcept
{
  return this->expr().size();
}

template <typename T>
inline const typename ExecutableExpression<T>::expression_type &ExecutableExpression<T>::expr() const noexcept
{
  return expr_;
}

template <typename T>
template <typename V>
inline PetscErrorCode ExecutableExpression<T>::Execute(ManagedMemory<V> &dest, PetscDeviceContext dctx) const noexcept
{
  PetscMemType    mtype = PETSC_MEMTYPE_DEVICE;
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  switch (dtype) {
  case PETSC_DEVICE_HOST:
    mtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_DEVICE_CUDA:
  case PETSC_DEVICE_HIP:
  case PETSC_DEVICE_SYCL:
    mtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_DEVICE_MAX:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid device type %d", static_cast<int>(dtype));
  }
  PetscCall(detail::ExecutePrepare(dctx, dtype, mtype, this->expr(), dest));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename V>
inline PetscErrorCode ExecutableExpression<T>::Execute(ManagedMemory<V> &dest) const noexcept
{
  PetscFunctionBegin;
  PetscCall(this->Execute(dest, this->dctx()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
PETSC_NODISCARD inline auto ident(T &&expr) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(MakeExpression<operators::identity>(std::forward<T>(expr)))

template <typename T, typename... E>
PETSC_NODISCARD inline std::pair<ManagedMemory<T> &, Expression<E...>> make_expr_pair(ManagedMemory<T> &dest, Expression<E...> &&expr) noexcept
{
  return {dest, std::move(expr)};
}

template <typename T, typename U = T>
PETSC_NODISCARD inline std::pair<ManagedMemory<T> &, Expression<operators::identity, const ManagedMemory<U> &>> make_expr_pair(ManagedMemory<T> &dest, const ManagedMemory<U> &src) noexcept
{
  return {dest, ident(src)};
}

} // namespace expr

// Force evaluation of the expression, but don't allow ManagedMemory (which is handled with a
// special overload)
template <typename T>
PETSC_NODISCARD inline util::enable_if_t<!util::is_instance<util::decay_t<T>, ManagedMemory>::value, expr::ExecutableExpression<util::remove_cvref_t<T>>> Eval(T &&expr, PetscDeviceContext dctx = nullptr) noexcept
{
  static_assert(expr::is_expression<T>::value, "");
  return {std::forward<T>(expr), dctx};
}

// wrap the managed type in a identity "expression" and pass it to the primary eval template
template <typename... T>
PETSC_NODISCARD inline auto Eval(const ManagedMemory<T...> &expr, PetscDeviceContext dctx = nullptr) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(Eval(expr::ident(expr), dctx))

template <typename... ManagedMemoryExprPair>
inline PetscErrorCode MultiEval(PetscDeviceContext dctx, ManagedMemoryExprPair &&...expressions) noexcept
{
  PetscMemType    mtype = PETSC_MEMTYPE_DEVICE;
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  switch (dtype) {
  case PETSC_DEVICE_HOST:
    mtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_DEVICE_CUDA:
  case PETSC_DEVICE_HIP:
  case PETSC_DEVICE_SYCL:
    mtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_DEVICE_MAX:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid device type %d", static_cast<int>(dtype));
  }
  PetscCall(expr::detail::MultiEvalExecutePrepare(dctx, dtype, mtype, std::forward<ManagedMemoryExprPair>(expressions)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace Petsc

#endif // PETSC_CPP_EXPR_EXECUTABLE_HPP
