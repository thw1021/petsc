#ifndef PETSC_CPP_EXPRESSION_HPP
#define PETSC_CPP_EXPRESSION_HPP

#include <petscdevice.h>

#include <petsc/private/cpp/expr/base.hpp>
#include <petsc/private/cpp/expr/constant.hpp>
#include <petsc/private/cpp/expr/frozen.hpp>

#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/tuple.hpp>

#include <cstddef> // std::size_t

namespace Petsc
{

template <typename>
class ManagedMemory;

namespace expr
{

// ==========================================================================================
// Expression
// ==========================================================================================

namespace detail
{

template <typename T, typename = void>
struct get_value_type {
  using type = T;
};

template <typename T>
struct get_value_type<T, util::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};

template <typename T>
using get_value_type_t = typename get_value_type<util::decay_t<T>>::type;

} // namespace detail

template <typename F, typename... T>
class Expression : public ExpressionBase<Expression<F, T...>> {
public:
  static_assert(sizeof...(T) > 0, "");
  using op_type         = F;
  using expression_type = std::tuple<T...>;
  using value_type      = util::common_type_t<detail::get_value_type_t<T>...>;
  using base_type       = ExpressionBase<Expression<F, T...>>;
  using size_type       = typename base_type::size_type;

  friend base_type;

  template <typename... Args>
  constexpr explicit Expression(op_type, Args &&...) noexcept;

  PetscErrorCode PrefetchBegin(PetscDeviceContext, PetscMemType, bool) const noexcept;
  PetscErrorCode PrefetchEnd(PetscDeviceContext, PetscMemType, bool) const noexcept;

  PETSC_NODISCARD const expression_type &expr() const noexcept;

  PETSC_NODISCARD auto Freeze(PetscMemType) const noexcept;

  PETSC_NODISCARD size_type size_impl_() const noexcept;

private:
  op_type         op_;
  expression_type expr_;

  template <typename U>
  static typename ManagedMemory<U>::const_pointer PtrOrExpr_(PetscMemType, const ManagedMemory<U> &) noexcept;

  template <typename... U>
  static auto PtrOrExpr_(PetscMemType, const Expression<U...> &) noexcept;

  template <typename U, util::enable_if_t<!is_expression<U>::value> * = nullptr>
  static ConstantExpression<util::decay_t<U>> PtrOrExpr_(PetscMemType, U &&) noexcept;

  template <std::size_t... Idx>
  auto Freeze_(PetscMemType, util::index_sequence<Idx...>) const noexcept;
};

// ==========================================================================================
// Expression -- Private API
// ==========================================================================================

template <typename F, typename... T>
template <typename U>
inline typename ManagedMemory<U>::const_pointer Expression<F, T...>::PtrOrExpr_(PetscMemType mtype, const ManagedMemory<U> &sub_expr) noexcept
{
  return PetscMemTypeHost(mtype) ? sub_expr.host_cdata() : sub_expr.device_cdata();
}

template <typename F, typename... T>
template <typename... U>
inline auto Expression<F, T...>::PtrOrExpr_(PetscMemType mtype, const Expression<U...> &expr) noexcept
{
  return expr.Freeze(mtype);
}

template <typename F, typename... T>
template <typename U, util::enable_if_t<!is_expression<U>::value> *>
inline ConstantExpression<util::decay_t<U>> Expression<F, T...>::PtrOrExpr_(PetscMemType, U &&scalar) noexcept
{
  return ConstantExpression<util::decay_t<U>>{std::forward<U>(scalar)};
}

template <typename F, typename... T>
template <std::size_t... Idx>
inline auto Expression<F, T...>::Freeze_(PetscMemType mtype, util::index_sequence<Idx...>) const noexcept
{
  return MakeFrozenExpression(op_, PtrOrExpr_(mtype, std::get<Idx>(this->expr()))...);
}

// ==========================================================================================
// Expression -- Public API
// ==========================================================================================

template <typename F, typename... T>
template <typename... Args>
inline constexpr Expression<F, T...>::Expression(op_type op, Args &&...args) noexcept : op_{std::move(op)}, expr_{std::forward<Args>(args)...}
{
}

template <typename F, typename... T>
inline const typename Expression<F, T...>::expression_type &Expression<F, T...>::expr() const noexcept
{
  return expr_;
}

namespace detail
{

template <bool begin>
struct Prefetcher {
  template <typename T>
  void operator()(const ManagedMemory<T> &expr) const noexcept
  {
    typename ManagedMemory<T>::value_type *ptr;

    PetscFunctionBegin;
    // kind of a wonky way of doing things, but you cannot early-return from tuple_for_each()
    // so instead we just abort on error.
    if (begin) {
      PetscCallAbort(PETSC_COMM_SELF, expr.GetArray(dctx, mtype, PETSC_MEMORY_ACCESS_READ, static_cast<PetscBool>(sync), &ptr));
    } else {
      PetscCallAbort(PETSC_COMM_SELF, expr.RestoreArray(dctx, mtype, PETSC_MEMORY_ACCESS_READ, static_cast<PetscBool>(sync), &ptr));
    }
    PetscFunctionReturnVoid();
  }

  template <typename... T>
  void operator()(const Expression<T...> &expr) const noexcept
  {
    PetscFunctionBegin;
    if (begin) {
      PetscCallAbort(PETSC_COMM_SELF, expr.PrefetchBegin(dctx, mtype, sync));
    } else {
      PetscCallAbort(PETSC_COMM_SELF, expr.PrefetchEnd(dctx, mtype, sync));
    }
    PetscFunctionReturnVoid();
  }

  PETSC_CONSTEXPR_14 void operator()(...) const noexcept { }

  PetscDeviceContext dctx;
  PetscMemType       mtype;
  bool               sync;
};

} // namespace detail

template <typename F, typename... T>
inline PetscErrorCode Expression<F, T...>::PrefetchBegin(PetscDeviceContext dctx, PetscMemType mtype, bool sync) const noexcept
{
  PetscFunctionBegin;
  util::tuple_for_each(this->expr(), detail::Prefetcher<true>{dctx, mtype, sync});
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename F, typename... T>
inline PetscErrorCode Expression<F, T...>::PrefetchEnd(PetscDeviceContext dctx, PetscMemType mtype, bool sync) const noexcept
{
  PetscFunctionBegin;
  util::tuple_for_each(this->expr(), detail::Prefetcher<false>{dctx, mtype, sync});
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename F, typename... T>
inline auto Expression<F, T...>::Freeze(PetscMemType mtype) const noexcept
{
  return Freeze_(mtype, util::index_sequence_for<T...>{});
}

namespace detail
{

struct SizeGetter {
  template <typename T>
  util::enable_if_t<is_expression<T>::value> operator()(const T &expr) noexcept
  {
    size = std::max(size, expr.size());
  }

  PETSC_CONSTEXPR_14 void operator()(...) const noexcept { }

  std::size_t size{};
};

} // namespace detail

template <typename F, typename... T>
inline typename Expression<F, T...>::size_type Expression<F, T...>::size_impl_() const noexcept
{
  return util::tuple_for_each(this->expr(), detail::SizeGetter{}).size;
}

namespace detail
{

template <typename F, typename... T>
struct expression_type {
  // clang-format off
  using type = Expression<
    util::remove_cvref_t<F>,
    util::conditional_t<
      std::is_lvalue_reference<T>::value,
      util::add_const_t<util::decay_t<T>> &,
      util::decay_t<T>
    >...
  >;
  // clang-format on
};

template <typename F, typename... T>
using expression_type_t = typename util::enable_if_t<has_expression<T...>::value, expression_type<F, T...>>::type;

} // namespace detail

template <typename F, typename... T>
inline constexpr detail::expression_type_t<F, T...> MakeExpression(T &&...args) noexcept
{
  return detail::expression_type_t<F, T...>{F{}, std::forward<T>(args)...};
}

} // namespace expr

} // namespace Petsc

#include <petsc/private/cpp/expr/operators.hpp>
#include <petsc/private/cpp/expr/executable.hpp>

#endif // PETSC_CPP_EXPRESSION_HPP
