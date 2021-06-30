#if !defined(PETSCOBJECTPOOL_HPP)
#define PETSCOBJECTPOOL_HPP

#include <petscsys.h>
#include <stack>

#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#include <type_traits>
#define PETSC_NULLPTR nullptr
#define PETSC_STATIC_ASSERT_BASE_CLASS(base_,derived_) static_assert(std::is_base_of<base_,derived_>::value,"")
#else
#define PETSC_NULLPTR NULL
#define PETSC_STATIC_ASSERT_BASE_CLASS(base_,derived_)
#endif

namespace Petsc {

// forward declare
template <typename T> struct allocator;

// generic allocator for interorperability with C "constructors" and "destructors"
template <typename T>
struct allocator {
  static PETSC_NODISCARD PetscErrorCode create(T*)  PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode destroy(T&) PETSC_NOEXCEPT;
};

// forward declare
template <typename T, class _Allocator> class objectPool;

// default implementation, use the petsc allocator
template <typename T, class _Allocator = allocator<T> > class objectPool;

// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Uses
// a stack to take advantage of LIFO for memory locallity. Registers all objects to be
// cleaned up on PetscFinalize()
template <typename T, class _Allocator>
class objectPool {
protected:
  PETSC_STATIC_ASSERT_BASE_CLASS(allocator<T>,_Allocator);
  typedef _Allocator allocator_t;
  std::stack<T>      _stack;
  PetscBool          _registered;

  // This exists to allow one to pass a function from C++ to PetscRegisterFinalize(). The
  // reasons for its construction are as follows:
  // 1. member-function-ness -> might need access to class internals, but I don't want a
  //                            bloated interface.
  // 2. static -> regular member functions have an implicit "this" pointer, no good for
  //              C. Static functions do not.
  static PETSC_NODISCARD PetscErrorCode finalize(void) PETSC_NOEXCEPT;

public:
  explicit PETSC_CONSTEXPR objectPool() PETSC_NOEXCEPT : _registered(PETSC_FALSE) {}

  PETSC_NODISCARD PetscErrorCode get(T&) PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode reclaim(T&) PETSC_NOEXCEPT;
};

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::get(T &obj) PETSC_NOEXCEPT
{
  PetscFunctionBegin;
  if (PetscUnlikely(!this->_registered)) {
    PetscErrorCode ierr;

    ierr = PetscRegisterFinalize(this->finalize);CHKERRQ(ierr);
    this->_registered = PETSC_TRUE;
  }
  try {
    if (this->_stack.empty()) {
      PetscErrorCode ierr;

      ierr = allocator_t::create(&obj);CHKERRQ(ierr);
    } else {
      obj = this->_stack.top();
      this->_stack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::reclaim(T &obj) PETSC_NOEXCEPT
{
  PetscFunctionBegin;
  try {
    this->_stack.push(obj);
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  obj = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* PETSCOBJECTPOOL_HPP */
