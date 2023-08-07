#ifndef PETSC_CPP_EXPR_OPERATORS_HPP
#define PETSC_CPP_EXPR_OPERATORS_HPP

#include <petscmath.h>
#include <petscmacros.h>
#include <petscdevice_cupm.h>

#include <petsc/private/cpp/macros.hpp>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/expr/expression.hpp>

#include <cmath> // math functions

namespace Petsc
{

namespace expr
{

namespace operators
{

template <bool b>
struct operator_base {
  static constexpr bool is_logical() noexcept { return b; }
};

} // namespace operators

#define PETSC_UNARY_OPERATOR_FUNCTOR(NAME, OP, IS_LOGICAL) \
  namespace operators \
  { \
  struct NAME : operator_base<IS_LOGICAL> { \
    template <typename T> \
    PETSC_HOSTDEVICE_INLINE_DECL constexpr auto operator()(const T &t) const PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(OP t) \
  }; \
  } /* namespace operators */ \
  template <typename T> \
  inline auto operator OP(T &&t) noexcept -> ::Petsc::expr::detail::template expression_type_t<::Petsc::expr::operators::NAME, T> \
  { \
    return ::Petsc::expr::MakeExpression<::Petsc::expr::operators::NAME>(std::forward<T>(t)); \
  }

#define PETSC_BINARY_OPERATOR_FUNCTOR(NAME, OP, IS_LOGICAL) \
  namespace operators \
  { \
  struct NAME : operator_base<IS_LOGICAL> { \
    template <typename T1, typename T2> \
    PETSC_HOSTDEVICE_INLINE_DECL constexpr auto operator()(const T1 &t1, const T2 &t2) const PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(t1 OP t2) \
  }; \
  } /* namespace operators */ \
  template <typename L, typename R> \
  inline auto operator OP(L &&lhs, R &&rhs) noexcept -> ::Petsc::expr::detail::template expression_type_t<::Petsc::expr::operators::NAME, L, R> \
  { \
    return ::Petsc::expr::MakeExpression<::Petsc::expr::operators::NAME>(std::forward<L>(lhs), std::forward<R>(rhs)); \
  }

PETSC_UNARY_OPERATOR_FUNCTOR(negate, -, false)
PETSC_UNARY_OPERATOR_FUNCTOR(logical_not, !, true)
PETSC_UNARY_OPERATOR_FUNCTOR(bitwise_not, ~, false)

PETSC_BINARY_OPERATOR_FUNCTOR(plus, +, false)
PETSC_BINARY_OPERATOR_FUNCTOR(minus, -, false)
PETSC_BINARY_OPERATOR_FUNCTOR(multiplies, *, false)
PETSC_BINARY_OPERATOR_FUNCTOR(divides, /, false)
PETSC_BINARY_OPERATOR_FUNCTOR(modulus, %, false)
// ASYNC TODO:
// overloading boolean operators makes them loose their short-circuit capabilities, do we
// really need this?
//
// PETSC_BINARY_OPERATOR_FUNCTOR(logical_or, ||)
// PETSC_BINARY_OPERATOR_FUNCTOR(logical_and, &&)
PETSC_BINARY_OPERATOR_FUNCTOR(bitwise_or, |, false)
PETSC_BINARY_OPERATOR_FUNCTOR(bitwise_and, &, false)
PETSC_BINARY_OPERATOR_FUNCTOR(bitwise_xor, ^, false)
PETSC_BINARY_OPERATOR_FUNCTOR(left_shift, <<, false)
PETSC_BINARY_OPERATOR_FUNCTOR(right_shift, >>, false)
PETSC_BINARY_OPERATOR_FUNCTOR(logical_less, <, true)
PETSC_BINARY_OPERATOR_FUNCTOR(logical_less_equal, <=, true)
PETSC_BINARY_OPERATOR_FUNCTOR(logical_greater, >, true)
PETSC_BINARY_OPERATOR_FUNCTOR(logical_greater_equal, >=, true)
PETSC_BINARY_OPERATOR_FUNCTOR(logical_equal_to, ==, true)
PETSC_BINARY_OPERATOR_FUNCTOR(logical_not_equal_to, !=, true)

#undef PETSC_UNARY_OPERATOR_FUNCTOR
#undef PETSC_BINARY_OPERATOR_FUNCTOR

// ASYNC TODO: trailing return type!
// https://stackoverflow.com/questions/19744324/decltype-and-mixing-adl-lookup-with-non-adl-lookup
#define PETSC_UNARY_MATH_FUNCTOR_(FNCTR_NAME, NAME) \
  namespace math \
  { \
  using std::NAME; \
  struct FNCTR_NAME { \
    template <typename T> \
    PETSC_HOSTDEVICE_INLINE_DECL constexpr auto operator()(T &&arg) const noexcept \
    { \
      using math::NAME; \
      return NAME(std::forward<T>(arg)); \
    } \
  }; \
  } /* namespace math */ \
  template <typename T> \
  inline auto NAME(T &&t) noexcept -> ::Petsc::expr::detail::template expression_type_t<::Petsc::expr::math::FNCTR_NAME, T> \
  { \
    return ::Petsc::expr::MakeExpression<::Petsc::expr::math::FNCTR_NAME>(std::forward<T>(t)); \
  }

#define PETSC_UNARY_MATH_FUNCTOR(NAME) PETSC_UNARY_MATH_FUNCTOR_(PetscConcat(NAME, _op), NAME)

#define PETSC_UNARY_MATH_FUNCTOR_NO_STD_(FNCTR_NAME, NAME, OP) \
  namespace math \
  { \
  struct FNCTR_NAME { \
    template <typename T> \
    PETSC_HOSTDEVICE_INLINE_DECL constexpr auto operator()(const T &arg) const PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(OP(arg)) \
  }; \
  } /* namespace math */ \
  template <typename T> \
  inline auto NAME(T &&t) noexcept -> ::Petsc::expr::detail::template expression_type_t<::Petsc::expr::math::FNCTR_NAME, T> \
  { \
    return ::Petsc::expr::MakeExpression<::Petsc::expr::math::FNCTR_NAME>(std::forward<T>(t)); \
  }

#define PETSC_UNARY_MATH_FUNCTOR_NO_STD(NAME, OP) PETSC_UNARY_MATH_FUNCTOR_NO_STD_(PetscConcat(NAME, _op), NAME, OP)

PETSC_UNARY_MATH_FUNCTOR(fabs)
PETSC_UNARY_MATH_FUNCTOR(abs)
PETSC_UNARY_MATH_FUNCTOR(sqrt)
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79700
//
// GCC has a longstanging bug where half the functions in <cmath> are not in namespace std,
// but instead in the global namespace, leading to confusing error messages:
//
// operators.hpp:131:26: error: no member named 'sqrtf' in namespace 'std'; did you mean
// 'sqrt'?
//
// According to the latest message (2023-07-27 10:42:27 UTC) there is a patch targeted for gcc
// 13.3, hopefully it is fixed by then?
#if defined(__GLIBCXX__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 130300
PETSC_UNARY_MATH_FUNCTOR_NO_STD(sqrtf, sqrtf)
#else
PETSC_UNARY_MATH_FUNCTOR(sqrtf)
#endif
PETSC_UNARY_MATH_FUNCTOR_NO_STD(sign, PetscSign)

// ASYNC TODO this is a HACK
#undef PetscSign
#define PetscSign(...) sign(__VA_ARGS__)

#undef PETSC_BINARY_MATH_FUNCTOR
#undef PETSC_BINARY_MATH_FUNCTOR_
#undef PETSC_BINARY_MATH_FUNCTOR_NO_STD
#undef PETSC_BINARY_MATH_FUNCTOR_NO_STD_

namespace operators
{

struct identity {
  template <typename T>
  PETSC_HOSTDEVICE_INLINE_DECL constexpr auto operator()(T &&t) const PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(std::forward<T>(t))
};

} // namespace operators

} // namespace expr

} // namespace Petsc

#endif // PETSC_CPP_EXPR_OPERATORS_HPP
