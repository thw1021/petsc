#if !defined(PETSCOBJECTPOOL_HPP)
#define PETSCOBJECTPOOL_HPP

#include <petscsys.h>
#include <stack>

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
#include <type_traits>
#define PETSC_STATIC_ASSERT_BASE_CLASS(base_,derived_,mess_) static_assert(std::is_base_of<base_,derived_>::value,mess_)
#else
#define PETSC_STATIC_ASSERT_BASE_CLASS(base_,derived_,mess_)
#endif

namespace Petsc {

// Allocator ABC for interoperability with C ctors and dtors.
template <typename T>
class Allocator {
public:
  typedef T value_type;

  PETSC_NODISCARD PetscErrorCode create(value_type*)  PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode destroy(value_type&) PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode reset(value_type&)   PETSC_NOEXCEPT;
};

// Default allocator that performs the bare minimum of petsc object creation and
// desctruction
template <typename T>
class CAllocator : Allocator<T>
{
public:
  typedef Allocator<T>                     allocator_t;
  typedef typename allocator_t::value_type value_type;

  PETSC_NODISCARD PetscErrorCode create(value_type *obj) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscNew(obj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode destroy(value_type &obj) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = (*obj->ops->destroy)(obj);CHKERRQ(ierr);
    ierr = PetscHeaderDestroy(&obj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode reset(value_type &obj) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = this->destroy(obj);CHKERRQ(ierr);
    ierr = this->create(&obj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
};

// Base class to object pool, defines helpful typedefs and stores the allocator instance
template <typename T, class _Allocator>
class objectPoolBase {
public:
  typedef _Allocator                       allocator_t;
  typedef typename allocator_t::value_type value_type;

protected:
  allocator_t _alloc;

  inline PETSC_NODISCARD allocator_t& getAllocator() PETSC_NOEXCEPT
  { return this->_alloc;}

  inline PETSC_NODISCARD const allocator_t& getAllocator() const PETSC_NOEXCEPT
  { return this->_alloc;}

  // default constructor
  PETSC_CONSTEXPR objectPoolBase() PETSC_NOEXCEPT_ARG(std::is_nothrow_default_constructible<allocator_t>::value) : _alloc() {}

  // const copy constructor
  explicit objectPoolBase(const allocator_t &alloc) : _alloc(alloc) {}

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  // move constructor
  explicit objectPoolBase(allocator_t &&alloc) PETSC_NOEXCEPT_ARG(std::is_nothrow_move_assignable<allocator_t>::value) : _alloc(std::move(alloc)) {}
#endif

  inline ~objectPoolBase()
  {
    PETSC_STATIC_ASSERT_BASE_CLASS(Allocator<value_type>,_Allocator,"Allocator type must be subclass of Petsc::Allocator");
  }
};

// default implementation, use the petsc c allocator
template <typename T, class _Allocator = CAllocator<T> > class objectPool;

// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Uses
// a stack to take advantage of LIFO for memory locallity. Registers all objects to be
// cleaned up on PetscFinalize()
template <typename T, class _Allocator>
class objectPool : private objectPoolBase<T,_Allocator> {
protected:
  typedef objectPoolBase<T,_Allocator> base_t;

public:
  typedef typename base_t::allocator_t allocator_t;
  typedef typename base_t::value_type  value_type;
  typedef std::stack<value_type>       stack_type;

protected:
  stack_type _stack;
  PetscBool  _registered;

  // This exists to allow one to pass a function from C++ to PetscRegisterFinalize(). The
  // reasons for its construction are as follows:
  // 1. member-function-ness -> might need access to class internals, but I don't want a
  //                            bloated interface.
  // 2. static -> regular member functions have an implicit "this" pointer, no good for
  //              C. Static functions do not.
  static PETSC_NODISCARD PetscErrorCode finalize(void) PETSC_NOEXCEPT;

public:
  // default constructor
  PETSC_CONSTEXPR objectPool() PETSC_NOEXCEPT_ARG(std::is_nothrow_default_constructible<allocator_t>::value) : _registered(PETSC_FALSE) {}

  // copy constructor
  objectPool(objectPool &other) PETSC_NOEXCEPT_ARG(std::is_nothrow_copy_constructible<stack_type>::value) : _stack(other._stack),_registered(other._registered) {}

  // const copy constructor
  objectPool(const objectPool &other) PETSC_NOEXCEPT_ARG(std::is_nothrow_copy_constructible<stack_type>::value) : _stack(other._stack),_registered(other._registered) {}

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  // move constructor
  objectPool(objectPool &&other) PETSC_NOEXCEPT_ARG(std::is_trivially_move_constructible<stack_type>::value) : _stack(std::move(other._stack)),_registered(std::move(other._registered)) {}
#endif

  // copy constructor with allocator
  explicit objectPool(const allocator_t &alloc) : base_t(alloc),_registered(PETSC_FALSE) {}

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  // move constructor with allocator
  explicit objectPool(allocator_t &&alloc) PETSC_NOEXCEPT_ARG(std::is_nothrow_move_constructible<allocator_t>::value) : base_t(std::move(alloc)),_registered(PETSC_FALSE) {}
#endif

  // Retrieve an object from the pool, if the pool is empty a new object is created instead
  PETSC_NODISCARD PetscErrorCode get(value_type&)      PETSC_NOEXCEPT;
  // Return an object to the pool, the object need not necessarily have been created by
  // the pool
  PETSC_NODISCARD PetscErrorCode reclaim(value_type&)  PETSC_NOEXCEPT;
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  PETSC_NODISCARD PetscErrorCode reclaim(value_type&&) PETSC_NOEXCEPT;
#endif

  // operators
  template <typename T_, class A_>
  friend PetscBool operator==(const objectPool<T_,A_>&,const objectPool<T_,A_>&);

  template <typename T_, class A_>
  friend PetscBool operator< (const objectPool<T_,A_>&,const objectPool<T_,A_>&);
};

template <typename T, class _Allocator>
inline PetscBool operator==(const objectPool<T,_Allocator> &l,const objectPool<T,_Allocator> &r)
{
  return static_cast<PetscBool>(l._stack == r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator< (const objectPool<T,_Allocator> &l, const objectPool<T,_Allocator> &r)
{
  return static_cast<PetscBool>(l._stack < r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator!=(const objectPool<T,_Allocator> &l, const objectPool<T,_Allocator> &r)
{
  return !(l._stack == r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator> (const objectPool<T,_Allocator> &l, const objectPool<T,_Allocator> &r)
{
  return r._stack < l._stack;
}

template <typename T, class _Allocator>
inline PetscBool operator>=(const objectPool<T,_Allocator> &l, const objectPool<T,_Allocator> &r)
{
  return !(l._stack < r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator<=(const objectPool<T,_Allocator> &l, const objectPool<T,_Allocator> &r)
{
  return !(r._stack < l._stack);
}

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::get(value_type &obj) PETSC_NOEXCEPT
{
  PetscFunctionBegin;
  if (PetscUnlikely(!this->_registered)) {
    PetscErrorCode ierr;

    ierr = PetscRegisterFinalize(this->finalize);CHKERRQ(ierr);
    this->_registered = PETSC_TRUE;
  }
  try {
    if (this->_stack.empty()) {
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
      // allows const allocator_t& to be used if allocator defines a const create
      auto           alloc = this->getAllocator();
#else
      allocator_t&   alloc = this->getAllocator();
#endif
      PetscErrorCode ierr;

      ierr = alloc.create(&obj);CHKERRQ(ierr);
    } else {
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
      obj = std::move(this->_stack.top());
#else
      obj = this->_stack.top();
#endif
      this->_stack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::reclaim(value_type &obj) PETSC_NOEXCEPT
{
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  // allows const allocator_t& to be used if allocator defines a const reset
  auto           alloc = this->getAllocator();
#else
  allocator_t&   alloc = this->getAllocator();
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = alloc.reset(obj);CHKERRQ(ierr);
  try {
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
    this->_stack.push(std::move(obj));
#else
    this->_stack.push(obj);
#endif
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  obj = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::reclaim(value_type &&obj) PETSC_NOEXCEPT
{
  // allows const allocator_t& to be used if allocator defines a const reset
  auto           alloc = this->getAllocator();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = alloc.reset(obj);CHKERRQ(ierr);
  try {
    this->_stack.push(std::move(obj));
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  obj = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}
#endif
} // namespace Petsc

#endif /* PETSCOBJECTPOOL_HPP */
