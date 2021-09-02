#ifndef PETSCTRAITHELPERS_HPP
#define PETSCTRAITHELPERS_HPP

#include <petsc/private/petscimpl.h>

#if defined(__cplusplus)

// A useful template to serve as a function wrapper factory. Given a function "foo" which
// you'd like to thinly wrap as "bar", simply doing:
//
// ALIAS_FUNCTION(bar,foo);
//
// essentially creates
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argn)
// { return foo(arg1,arg2,...,argn);}
//
// for you. You may then call bar exactly as you would foo.
#if PetscDefined(HAVE_CXX_DIALECT_CXX14) && __cplusplus >= 201402L
// decltype(auto) is c++14
#define PETSC_ALIAS_FUNCTION(Alias_,Original_)                          \
  template <typename... Args>                                           \
  PETSC_NODISCARD decltype(auto) Alias_(Args&&... args)                 \
  { return Original_(std::forward<Args>(args)...);}
#else
#define PETSC_ALIAS_FUNCTION(Alias_,Original_)                          \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto Alias_(Args&&... args)                           \
    -> decltype(Original_(std::forward<Args>(args)...))                 \
  { return Original_(std::forward<Args>(args)...);}
#endif // PetscDefined(HAVE_CXX_DIALECT_CXX14)

#if __cplusplus >= 201402L // (c++14)
using std::enable_if_t;
using std::decay_t;
using std::remove_pointer_t;
using std::remove_reference_t;
using std::remove_const;
using std::remove_volatile;
using std::remove_cv_t;
using std::conditional_t;
using std::tuple_element_t;
using std::index_sequence;
using std::make_index_sequence;
#else
template <bool B, typename T = void> using enable_if_t   = typename std::enable_if<B,T>::type;
template< bool B, class T, class F > using conditional_t = typename std::conditional<B,T,F>::type;
template <std::size_t I, class T> using tuple_element_t  = typename std::tuple_element<I,T>::type;
template <class T> using decay_t            = typename std::decay<T>::type;
template <class T> using remove_pointer_t   = typename std::remove_pointer<T>::type;
template <class T> using remove_reference_t = typename std::remove_reference<T>::type;
template <class T> using remove_const_t     = typename std::remove_const<T>::type;
template <class T> using remove_volatile_t  = typename std::remove_volatile<T>::type;
template <class T> using remove_cv_t        = typename std::remove_cv<T>::type;

// forward declare
template <std::size_t... Idx> struct index_sequence;

namespace detail {

template <class LeftSequence, class RightSequence> struct merge_sequences;

template <std::size_t... I1, std::size_t... I2>
struct merge_sequences<index_sequence<I1...>, index_sequence<I2...>>
  : index_sequence<I1...,(sizeof...(I1)+I2)...>
{ };

} // namespace detail

template <std::size_t N>
struct make_index_sequence
  : detail::merge_sequences<typename make_index_sequence<N/2>::type,
                            typename make_index_sequence<N-N/2>::type>
{ };

template <std::size_t... Idx>
struct index_sequence
{
  using type       = index_sequence;
  using value_type = std::size_t;

  static constexpr std::size_t size() noexcept { return sizeof...(Idx); }
};

template <> struct make_index_sequence<0> : index_sequence<>  { };
template <> struct make_index_sequence<1> : index_sequence<0> { };
#endif // __cplusplus >= 201402L (c++14)

#if __cplusplus >= 202002L // (c++20)
using remove_cvref_t = std::remove_cvref_t;
#else
template <class T> struct remove_cvref { typedef remove_cv_t<remove_reference_t<T>> type; };
template <class T> using remove_cvref_t = typename remove_cvref<T>::type;
#endif // __cplusplus >= 202002L (c++20)

namespace detail {

template <typename T>
struct strip_function
{
  typedef remove_pointer_t<decay_t<T>> type;
};

template <typename T>
struct is_callable_function : std::is_function<typename strip_function<T>::type>
{ };

template <typename T>
struct is_callable_class
{
  typedef char yay;
  typedef long nay;

  template <typename C> static yay test(decltype(&remove_reference_t<C>::operator()));
  template <typename C> static nay test(...);

public:
  static constexpr bool value = sizeof(test<T>(0)) == sizeof(yay);
  //enum { value = sizeof(test<T>(0)) == sizeof(yay) };
};

template <bool... bs>
struct all_true
{
  template<bool...> struct bool_pack;

  //if any are false, they'll be shifted in the second version, so types won't match
  static constexpr bool value = std::is_same<bool_pack<bs...,true>,bool_pack<true,bs...>>::value;
};

template <typename... Ts>
using all_true_exp = all_true<Ts::value...>;

} // namespace detail

template <typename T, typename... Args>
struct is_invocable
{
  template <typename U>
  static auto test(U *p) -> decltype((*p)(std::declval<Args>()...),void(),std::true_type());

  template <typename U>
  static auto test(...) -> decltype(std::false_type());

  static constexpr bool value = decltype(test<decay_t<T>>(0))::value;
};

template <typename T, typename... Ts>
using all_same = detail::all_true<std::is_same<T,Ts>::value...>;

template <typename T>
struct is_callable : conditional_t<
  std::is_class<T>::value,
  detail::is_callable_class<T>,
  detail::is_callable_function<T>
  >
{ };

// lambda
template <typename T>
struct function_traits : function_traits<decltype(&remove_reference_t<T>::operator())>
{ };

// function reference
template <typename R, typename... Args>
struct function_traits<R(&)(Args...)> : function_traits<R(Args...)>
{ };

// function pointer
template <typename R, typename... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)>
{ };

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)> : function_traits<R(Args...)>
{ };

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> : function_traits<R(Args...)>
{ };

// member object pointer
template <typename C, typename R>
struct function_traits<R(C::*)> : function_traits<R()>
{ };

// std::function
template <typename T>
struct function_traits<std::function<T>> : function_traits<T>
{ };

// generic function form
template <typename R, typename... Args>
struct function_traits<R(Args...)>
{
  using return_type = R;

  // arity is the number of arguments.
  static constexpr std::size_t arity = sizeof...(Args);

  // template <std::size_t N, enable_if_t<(N<arity)&&(N>=0)>* = nullptr>
  // using argument = tuple_element_t<N,std::tuple<Args...,void>>;
  template <std::size_t N>
  struct argument
  {
    static_assert((N < arity) && (N >= 0), "error: invalid parameter index.");
    using type = tuple_element_t<N,std::tuple<Args...,void>>;
  };
};

#define IDENTITY(X) X

#define NAMESPACE_QUALIFY(NS,NAME) IDENTITY(NS::NAME)

#define STRINGIZE(arg)  STRINGIZE1(arg)
#define STRINGIZE1(arg) STRINGIZE2(arg)
#define STRINGIZE2(arg) IDENTITY(#arg)

#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2)  IDENTITY(arg1##arg2)

#define FOR_EACH_1(WHAT, x) WHAT(x);
#define FOR_EACH_2(WHAT, x, ...) WHAT(x);FOR_EACH_1(WHAT,__VA_ARGS__);
#define FOR_EACH_3(WHAT, x, ...) WHAT(x);FOR_EACH_2(WHAT,__VA_ARGS__);
#define FOR_EACH_4(WHAT, x, ...) WHAT(x);FOR_EACH_3(WHAT,__VA_ARGS__);
#define FOR_EACH_5(WHAT, x, ...) WHAT(x);FOR_EACH_4(WHAT,__VA_ARGS__);
#define FOR_EACH_6(WHAT, x, ...) WHAT(x);FOR_EACH_5(WHAT,__VA_ARGS__);
#define FOR_EACH_7(WHAT, x, ...) WHAT(x);FOR_EACH_6(WHAT,__VA_ARGS__);
#define FOR_EACH_8(WHAT, x, ...) WHAT(x);FOR_EACH_7(WHAT,__VA_ARGS__);

#define FOR_EACH_WITH_1(WITH, WHAT, x) WHAT(WITH,x);
#define FOR_EACH_WITH_2(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_1(WITH,WHAT,__VA_ARGS__);
#define FOR_EACH_WITH_3(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_2(WITH,WHAT,__VA_ARGS__);
#define FOR_EACH_WITH_4(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_3(WITH,WHAT,__VA_ARGS__);
#define FOR_EACH_WITH_5(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_4(WITH,WHAT,__VA_ARGS__);
#define FOR_EACH_WITH_6(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_5(WITH,WHAT,__VA_ARGS__);
#define FOR_EACH_WITH_7(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_6(WITH,WHAT,__VA_ARGS__);
#define FOR_EACH_WITH_8(WITH, WHAT, x, ...) WHAT(WITH,x);FOR_EACH_WITH_7(WITH,WHAT,__VA_ARGS__);

#define FOR_EACH_NARG(...)  FOR_EACH_NARG_(__VA_ARGS__,FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define FOR_EACH_(N, WHAT, x, ...) CONCATENATE(FOR_EACH_, N)(WHAT, x, __VA_ARGS__)
#define FOR_EACH(WHAT, x, ...)                                          \
  do {                                                                  \
    FOR_EACH_(FOR_EACH_NARG(x, __VA_ARGS__), WHAT, x, __VA_ARGS__);     \
  } while (0)

#define FOR_EACH_WITH_(N, WITH, WHAT, x, ...)                           \
  do {                                                                  \
    CONCATENATE(FOR_EACH_WITH_, N)(WITH, WHAT, x, __VA_ARGS__);         \
  } while (0)


#define FOR_EACH_WITH(WITH, WHAT, x, ...)                               \
  do {                                                                  \
    FOR_EACH_WITH_(FOR_EACH_NARG(x,__VA_ARGS__),                        \
                   WITH,WHAT,x,__VA_ARGS__);                            \
  } while (0)

#define ENUM_QUALIFY(Enum) NAMESPACE_QUALIFY(CONCATENATE(Enum,Wrapper),Enum)

#define PRINT_CASE_LABEL(EnumName,LabelName)                            \
  do {                                                                  \
    case NAMESPACE_QUALIFY(ENUM_QUALIFY(EnumName),LabelName):           \
      return STRINGIZE(NAMESPACE_QUALIFY(EnumName,LabelName));          \
  } while (0)

#define PETSC_CXX_ENUM_WRAPPER_DECLARE(EnumName, UnderlyingType, ...)   \
  class EnumName##Wrapper                                               \
  {                                                                     \
  public:                                                               \
  enum class EnumName : UnderlyingType { __VA_ARGS__ };                 \
                                                                        \
  PETSC_NODISCARD PETSC_STATIC_INLINE                                   \
  const char * __EnumName##ToString(ENUM_QUALIFY(EnumName) name)        \
    {                                                                   \
      switch (name) {                                                   \
        FOR_EACH_WITH(EnumName,PRINT_CASE_LABEL,__VA_ARGS__);           \
      }                                                                 \
    }                                                                   \
                                                                        \
  private:                                                              \
  friend std::ostream                                                   \
  &operator<<(std::ostream &strm, ENUM_QUALIFY(EnumName) name)          \
    {                                                                   \
      return strm << __EnumName##ToString(name);                        \
    }                                                                   \
  };                                                                    \
  PETSC_NODISCARD PETSC_STATIC_INLINE                                   \
  const char *EnumName##ToString(ENUM_QUALIFY(EnumName) name)           \
  {                                                                     \
    return NAMESPACE_QUALIFY(CONCATENATE(EnumName, Wrapper),            \
                             CONCATENATE(__EnumName, ToString))(name);  \
  }                                                                     \
  using EnumName = ENUM_QUALIFY(EnumName);

#endif /* __cplusplus */

#endif /* PETSCTRAITHELPERS_HPP */
