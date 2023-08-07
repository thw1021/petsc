#ifndef PETSC_CPP_EXPR_FROZEN_HPP
#define PETSC_CPP_EXPR_FROZEN_HPP

#include <petscdevice_cupm.h>

#include <petsc/private/cpp/tuple.hpp>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>

#include <cstddef> // std::size_t

namespace Petsc
{

namespace expr
{

// ==========================================================================================
// FrozenExpression
//
// Since the lifetimes of a device lambda may outlast its captured elements, they need to be
// captured by value. This is not practical for 2 reasons:
//
// 1. An expression could contain data still on the host
// 2. It may be prohibitively expensive to copy-construct expressino components (such as
//    ManagedType)
//
// We therefore need to "freeze" the expression into its most basic form. FrozenExpression is
// that form. It contains the device operator, and either a tuple of sub-frozen expressions, or
// the raw pointers themselves.
//
// The access operator[] indexes the arrays themselves.
// ==========================================================================================

template <typename F, typename... T>
class FrozenExpression {
public:
  using size_type       = std::size_t;
  using operator_type   = F;
  using expression_type = std::tuple<T...>;
  using value_type      = util::invoke_result_t<operator_type, decltype(std::declval<T>()[0])...>;

  FrozenExpression() = delete;

  // Need to define all of these inline, otherwise hipcc chokes on it with
  // error: __host__ function 'XXX' cannot overload __host__ __device__ function 'XXX'
  template <typename... U>
  constexpr explicit FrozenExpression(operator_type op, U &&...expr) noexcept : op_{std::move(op)}, expr_{std::forward<U>(expr)...}
  {
  }

  PETSC_HOSTDEVICE_INLINE_DECL value_type operator[](size_type idx) const noexcept { return at_(util::index_sequence_for<T...>{}, idx); }

private:
  operator_type   op_;
  expression_type expr_;

  template <std::size_t... Idx>
  PETSC_HOSTDEVICE_INLINE_DECL value_type at_(util::index_sequence<Idx...>, size_type idx) const noexcept
  {
    return op_(std::get<Idx>(expr_)[idx]...);
  }
};

namespace detail
{

template <typename T>
struct frozen_type {
  // clang-format off
  using type = util::conditional_t<
    std::is_pointer<T>::value,
    const util::remove_pointer_t<T> *PETSC_RESTRICT,
    T
  >;
  // clang-format on
};

template <typename T>
using frozen_type_t = typename frozen_type<util::decay_t<T>>::type;

} // namespace detail

template <typename... T>
inline FrozenExpression<detail::frozen_type_t<T>...> MakeFrozenExpression(T &&...args) noexcept
{
  return FrozenExpression<detail::frozen_type_t<T>...>{std::forward<T>(args)...};
}

} // namespace expr

} // namespace Petsc

#endif // PETSC_CPP_EXPR_FROZEN_HPP
