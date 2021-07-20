#if !defined(PETSCOBJECTPOOL_HPP)
#define PETSCOBJECTPOOL_HPP

#include <petscsys.h>
#include <stack>
#include <functional>

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
  PETSC_NODISCARD PetscErrorCode finalize(void)       PETSC_NOEXCEPT;

protected:
  // make the constructor protected, this forces this class to be derived from to ever be
  // instantiated
  Allocator() {}
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

  PETSC_NODISCARD PetscErrorCode finalize(void) const PETSC_NOEXCEPT { return 0;}
};

// Base class to object pool, defines helpful typedefs and stores the allocator instance
template <typename T, class _Allocator>
class objectPoolBase {
public:
  typedef _Allocator                       allocator_t;
  typedef typename allocator_t::value_type value_type;

protected:
  allocator_t _alloc;

  inline PETSC_NODISCARD allocator_t& __getAllocator() PETSC_NOEXCEPT
  { return this->_alloc;}

  inline PETSC_NODISCARD const allocator_t& __getAllocator() const PETSC_NOEXCEPT
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
class objectPool : objectPoolBase<T,_Allocator> {
protected:
  typedef objectPoolBase<T,_Allocator> base_t;

public:
  typedef typename base_t::allocator_t allocator_t;
  typedef typename base_t::value_type  value_type;
  typedef std::stack<value_type>       stack_type;

protected:
  stack_type _stack;
  PetscBool  _registered = PETSC_FALSE;

private:
  static PETSC_NODISCARD PetscErrorCode __staticFinalizer(void*) PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode __finalizer(void) PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode __registerFinalize(void) PETSC_NOEXCEPT;
public:
  // default constructor
  PETSC_CONSTEXPR objectPool() PETSC_NOEXCEPT_ARG(std::is_nothrow_default_constructible<allocator_t>::value) {}

  // copy constructor
  objectPool(objectPool &other) PETSC_NOEXCEPT_ARG(std::is_nothrow_copy_constructible<stack_type>::value) : _stack(other._stack),_registered(other._registered) {}

  // const copy constructor
  objectPool(const objectPool &other) PETSC_NOEXCEPT_ARG(std::is_nothrow_copy_constructible<stack_type>::value) : _stack(other._stack),_registered(other._registered) {}

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  // move constructor
  objectPool(objectPool &&other) PETSC_NOEXCEPT_ARG(std::is_nothrow_move_constructible<stack_type>::value) : _stack(std::move(other._stack)),_registered(std::move(other._registered)) {}
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
PetscErrorCode objectPool<T,_Allocator>::__staticFinalizer(void *obj) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = static_cast<objectPool<T,_Allocator>*>(obj)->__finalizer();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::__finalizer(void) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    while (!this->_stack.empty()) {
      ierr = this->__getAllocator().destroy(this->_stack.top());CHKERRQ(ierr);
      this->_stack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  ierr = this->__getAllocator().finalize();CHKERRQ(ierr);
  this->_registered = PETSC_FALSE;
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::__registerFinalize(void) PETSC_NOEXCEPT
{
  PetscFunctionBegin;
  if (PetscUnlikely(!this->_registered)) {
    PetscErrorCode ierr;
    PetscContainer contain;

    /* use a PetscContainer as a form of thunk, it holds not only a pointer to this but
       also the pointer to the static member function, which just converts the thunk back
       to this. none of this would be needed if PetscRegisterFinalize() just took a void*
       itself though...  */
    ierr = PetscContainerCreate(PETSC_COMM_SELF,&contain);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(contain,this);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(contain,this->__staticFinalizer);CHKERRQ(ierr);
    ierr = PetscObjectRegisterDestroy((PetscObject)contain);CHKERRQ(ierr);
    this->_registered = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::get(value_type &obj) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = this->__registerFinalize();CHKERRQ(ierr);
  try {
    if (this->_stack.empty()) {
      ierr = this->__getAllocator().create(&obj);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikely(!this->_registered)) {
    /* this is necessary if an object is "reclaimed" within another PetscFinalize()
       registered cleanup after this object pool has returned from it's finalizer. In
       this case, instead of pushing onto the stack we just destroy the object directly */
    ierr = this->__getAllocator().destroy(obj);CHKERRQ(ierr);
  } else {
    ierr = this->__getAllocator().reset(obj);CHKERRQ(ierr);
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
  }
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
template <typename T, class _Allocator>
PetscErrorCode objectPool<T,_Allocator>::reclaim(value_type &&obj) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikely(!this->_registered)) {
    /* this is necessary if an object is "reclaimed" within another PetscFinalize()
       registered cleanup after this object pool has returned from it's finalizer. In
       this case, instead of pushing onto the stack we just destroy the object directly */
    ierr = this->__getAllocator().destroy(obj);CHKERRQ(ierr);
  } else {
    // allows const allocator_t& to be used if allocator defines a const reset
    ierr = this->__getAllocator().reset(obj);CHKERRQ(ierr);
    try {
      this->_stack.push(std::move(obj));
    } catch (std::exception const &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
    }
    obj = PETSC_NULLPTR;
  }
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_CXX_DIALECT_CXX11 */

} /* namespace Petsc */

#endif /* PETSCOBJECTPOOL_HPP */
