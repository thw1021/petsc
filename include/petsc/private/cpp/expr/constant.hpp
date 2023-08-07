#ifndef PETSC_CPP_EXPR_CONSTANT_HPP
#define PETSC_CPP_EXPR_CONSTANT_HPP

#include <petscdevice_cupm.h> // PETSC_HOSTDEVICE_INLINE_DECL

#include <cstddef> // std::size_t

namespace Petsc
{

namespace expr
{

// ==========================================================================================
// ConstantExpression
//
// A simple wrapper to make a single value have an operator[].
// ==========================================================================================

template <typename T>
class ConstantExpression {
public:
  using value_type = T;

  constexpr ConstantExpression() noexcept = default;

  PETSC_HOSTDEVICE_INLINE_DECL constexpr explicit ConstantExpression(const value_type &) noexcept;

  PETSC_HOSTDEVICE_INLINE_DECL constexpr const value_type &operator[](std::size_t) const noexcept;

private:
  value_type v_{};
};

// ==========================================================================================
// ConstantExpression -- Public API
// ==========================================================================================

template <typename T>
inline constexpr ConstantExpression<T>::ConstantExpression(const value_type &v) noexcept : v_{v}
{
}

template <typename T>
inline constexpr const typename ConstantExpression<T>::value_type &ConstantExpression<T>::operator[](std::size_t) const noexcept
{
  return v_;
}

} // namespace expr

} // namespace Petsc

#endif // PETSC_CPP_EXPR_CONSTANT_HPP
