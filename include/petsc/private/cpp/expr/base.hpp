#ifndef PETSC_CPP_EXPR_BASE_HPP
#define PETSC_CPP_EXPR_BASE_HPP

#include <petscmacros.h> // PETSC_NODISCARD

#include <petsc/private/cpp/type_traits.hpp> // disjunction, decay_t, is_crtp_base_of

#include <cstddef> // std::size_t

namespace Petsc
{

namespace expr
{

// ==========================================================================================
// ExpressionBase
// ==========================================================================================

template <typename D>
class ExpressionBase {
public:
  using size_type = std::size_t;

  PETSC_NODISCARD size_type size() const noexcept;
};

// ==========================================================================================
// ExpressionBase -- Public API
// ==========================================================================================

template <typename D>
inline typename ExpressionBase<D>::size_type ExpressionBase<D>::size() const noexcept
{
  return static_cast<const D *>(this)->size_impl_();
}

template <typename T>
struct is_expression : util::is_crtp_base_of<ExpressionBase, util::decay_t<T>> { };

template <typename... T>
struct has_expression : util::disjunction<is_expression<T>...> { };

} // namespace expr

} // namespace Petsc

#endif // PETSC_CPP_EXPR_BASE_HPP
