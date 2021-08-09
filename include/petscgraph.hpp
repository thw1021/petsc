#ifndef PETSCGRAPH_HPP
#define PETSCGRAPH_HPP

#include <petsc/private/petscimpl.h>

#if defined(__cplusplus)

#include <vector>
#include <unordered_map>
#include <stack>
#include <iostream>
#include <functional>

namespace Petsc {

namespace detail {

namespace impl {

template <typename T>
struct is_callable_class
{
  typedef char yay;
  typedef long nay;

  template <typename C> static yay test(decltype(&C::operator()));
  template <typename C> static nay test(...);

public:
  enum { value = sizeof(test<T>(0)) == sizeof(yay) };
};

template <typename T>
struct strip_function
{
  typedef typename std::remove_pointer<typename std::decay<T>::type>::type type;
};

template <typename T>
struct is_callable_function : std::is_function<typename strip_function<T>::type>
{ };

template <bool... bs>
struct all_true
{
  template<bool...> struct bool_pack;

  //if any are false, they'll be shifted in the second version, so types won't match
  static constexpr bool value = std::is_same<bool_pack<bs...,true>,bool_pack<true,bs...>>::value;
};

template <typename... Ts>
using all_true_exp = all_true<Ts::value...>;

} // namespace impl

template <bool B, typename T = void>
#if __cplusplus >= 201402L // c++14
using enable_if_t = std::enable_if_t<B,T>;
#else
using enable_if_t = typename std::enable_if<B,T>::type;
#endif

template <typename T> struct function_traits;

// generic function form
template <typename R, typename... Args>
struct function_traits<R(Args...)>
{
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  using return_type = R;

  template <std::size_t N>
  struct arg
  {
    static_assert((N < arity) && (N >= 0), "error: invalid parameter index.");
    using type = typename std::tuple_element<N,std::tuple<Args...,void>>::type;
  };
};

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
struct function_traits<std::function<T>> : public function_traits<T>
{ };

// lvalue reference
template <typename T>
struct function_traits<T&> : public function_traits<T>
{ };

// const lvalue reference
template <typename T>
struct function_traits<const T&> : public function_traits<T>
{ };

// rvalue reference
template <typename T>
struct function_traits<T&&> : public function_traits<T>
{ };

// lambda
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())>
{ };

// // functor
// template <typename T>
// struct function_traits
// {
// private:
//   using call_type = function_traits<decltype(&T::operator())>;

// public:
//   using return_type = typename call_type::return_type;

//   static constexpr std::size_t arity = call_type::arity;

//   template <std::size_t N>
//   struct arg
//   {
//     static_assert(N < arity, "error: invalid parameter index.");
//     using type = typename call_type::template arg<N+1>::type;
//   };
// };

template <class T>
struct is_callable : std::conditional<
  std::is_class<T>::value,
  impl::is_callable_class<T>,
  impl::is_callable_function<T>
  >::type
{ };

template <typename T, typename... Ts>
using all_same = impl::all_true_exp<std::is_same<T,Ts>...>;

template<int ...> struct IntegerSequence { };

template<int N, int ...S> struct SequenceGenerator : SequenceGenerator<N-1, N-1, S...> { };

template<int ...S> struct SequenceGenerator<0, S...> : IntegerSequence<S...> { };

template <int I = 0, typename Tf, typename... Tp, enable_if_t<I == sizeof...(Tp)>* = nullptr>
inline constexpr PetscErrorCode forEachInTuple(Tf &&fn, std::tuple<Tp...> &tuple) { return 0;}

template <int I = 0, typename Tf, typename... Tp, enable_if_t<(I < sizeof...(Tp))>* = nullptr>
inline PetscErrorCode forEachInTuple(Tf &&fn, std::tuple<Tp...> &tuple)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = fn(std::get<I>(tuple));CHKERRQ(ierr);
  ierr = forEachInTuple<I+1>(fn,tuple);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace detail

class CallNode
{
private:
  struct CallableBase // ABC for type-erasing the callable
  {
  public:
    using ptr_t = std::unique_ptr<const CallableBase>;

    virtual ~CallableBase()                        = default;
    virtual PetscErrorCode operator()(void*) const = 0;
    ptr_t clone() const { return ptr_t{cloneDerived()};}

  protected:
    virtual CallableBase* cloneDerived() const = 0;
  };

  template <typename... Args>
  struct WrappedCallable final : CallableBase // actual callable
  {
  public:
    using signature_t = PetscErrorCode(void*,Args...);
    using function_t  = std::function<signature_t>;
    using argPack_t   = std::tuple<Args...>;
    using self_t = WrappedCallable<Args...>;
    using base_t = CallableBase;

  private:
    const function_t _functor;
    argPack_t        _params;

    template <int... S>
    PetscErrorCode __callFunc(void *ctx, detail::IntegerSequence<S...>) const
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = _functor(ctx,std::get<S>(_params)...);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

  protected:
    self_t* cloneDerived() const override final { return new self_t{*this};}

  public:
    template <typename T>
    WrappedCallable(T &&fn, argPack_t &&args)
      : _functor{std::forward<T>(fn)}, _params{std::forward<argPack_t>(args)}
    { }

    PetscErrorCode operator()(void *ctx) const override final
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = __callFunc(ctx,detail::SequenceGenerator<sizeof...(Args)>());CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  };

  template <typename... Args>
  struct NativeCallable final : CallableBase // actual callable
  {
  public:
    using signature_t = PetscErrorCode(Args...);
    using function_t  = std::function<signature_t>;
    using argPack_t   = std::tuple<Args...>;
    using self_t = NativeCallable<Args...>;
    using base_t = CallableBase;

  private:
    const function_t _functor;
    argPack_t        _params;

    template <int... S>
    PetscErrorCode __callFunc(detail::IntegerSequence<S...>) const
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = _functor(std::get<S>(_params)...);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

  protected:
    self_t* cloneDerived() const override final { return new self_t{*this};}

  public:
    template <typename T>
    NativeCallable(T &&fn, argPack_t &&args)
      : _functor{std::forward<T>(fn)}, _params{std::forward<argPack_t>(args)}
    { }

    PetscErrorCode operator()(void *ctx) const override final
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = __callFunc(detail::SequenceGenerator<sizeof...(Args)>());CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  };

protected:
  CallableBase::ptr_t    _functor = nullptr;
  std::vector<CallNode*> _inedges;
  std::vector<CallNode*> _outedges;
  PetscInt               _stream = 0;
  const PetscInt         _id;

  static PetscInt       counter;

public:
  // Default constructor
  CallNode() : _id{counter++} { }

  // Copy constructor
  CallNode(const CallNode &other)
    : _functor{other._functor->clone()}, _inedges{other._inedges}, _outedges{other._outedges},
      _id{counter++}
  { }

  // Move constructor
  CallNode(CallNode &&other) noexcept = default;

  // Templated constructor with (optional) tuple of arguments, only enabled if first
  // argument is NOT a void *
  template <typename T, typename ...Args,
            detail::enable_if_t<
              !std::is_same<
                typename detail::function_traits<T>::template arg<0>::type,
                void*
                >::value
              >* = nullptr>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor{new NativeCallable<Args...>{std::forward<T>(f),std::forward<std::tuple<Args...>>(args)}},
      _id{counter++}
  { static_assert(sizeof...(Args), "need args for native call");}

  // Templated constructor with (optional) tuple of arguments, only enabled if first
  // argument is a void *
  template <typename T, typename ...Args,
            detail::enable_if_t<
              std::is_same<
                typename detail::function_traits<T>::template arg<0>::type,
                void*
                >::value
              >* = nullptr>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor{new WrappedCallable<Args...>{std::forward<T>(f),std::forward<std::tuple<Args...>>(args)}},
      _id{counter++}
  { }

  // Templated constructor with bare arguments
  template <typename T, typename ...Args>
  CallNode(T &&f, Args&&... args)
    : CallNode{std::forward<T>(f),std::make_tuple(std::forward<Args>(args)...)}
  { }

  // Destructor
  ~CallNode() {std::cout<<"node dtor "<<_id<<std::endl;}

  // Copy assignment operator
  CallNode& operator=(const CallNode&);

  // Move assignment operator
  CallNode& operator=(CallNode &&) noexcept;

  // Call operator
  PetscErrorCode operator()(void*) const;

  // Private member accessors
  PetscInt id() const { return _id;}
  const std::vector<CallNode*>& inedges()  const { return _inedges; }
  const std::vector<CallNode*>& outedges() const { return _outedges;}
  PetscInt stream() const { return _stream;}
  PetscErrorCode setStream(PetscInt stream) { _stream = stream; return 0;}

  // Order enforcement
  PetscErrorCode before(CallNode&);
  template <typename... Args,
            detail::enable_if_t<detail::all_same<CallNode*,Args...>::value>* = nullptr>
  PetscErrorCode before(std::tuple<Args...> &others);
  PetscErrorCode after(CallNode&);
  template <typename... Args,
            detail::enable_if_t<detail::all_same<CallNode*,Args...>::value>* = nullptr>
  PetscErrorCode after(std::tuple<Args...> &others);
};

CallNode& CallNode::operator=(const CallNode &other)
{
  PetscFunctionBegin;
  if (this != &other) {
    if (other._functor) {_functor = other._functor->clone();}
    _inedges  = other._inedges;
    _outedges = other._outedges;
  }
  PetscFunctionReturn(*this);
}

CallNode& CallNode::operator=(CallNode &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    _functor  = std::move(other._functor);
    _inedges  = std::move(other._inedges);
    _outedges = std::move(other._outedges);
  }
  PetscFunctionReturn(*this);
}

PetscErrorCode CallNode::operator()(void *ctx) const
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  std::cout<<"- node "<<_id<<" running on stream "<<_stream<<'\n';
  ierr = (*_functor)(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CallNode::before(CallNode &other)
{
  PetscFunctionBegin;
  std::cout<<_id<<" before "<<other._id<<std::endl;
  CHKERRCXX(other._inedges.push_back(this));
  CHKERRCXX(_outedges.push_back(&other));
  PetscFunctionReturn(0);
}

template <typename... Args, detail::enable_if_t<detail::all_same<CallNode*,Args...>::value>*>
PetscErrorCode CallNode::before(std::tuple<Args...> &others)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = detail::forEachInTuple([this](CallNode *other){return before(*other);},others);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CallNode::after(CallNode &other)
{
  PetscFunctionBegin;
  std::cout<<_id<<" after "<<other._id<<std::endl;
  CHKERRCXX(_inedges.push_back(&other));
  CHKERRCXX(other._outedges.push_back(this));
  PetscFunctionReturn(0);
}

template <typename... Args, detail::enable_if_t<detail::all_same<CallNode*,Args...>::value>*>
PetscErrorCode CallNode::after(std::tuple<Args...> &others)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = detail::forEachInTuple([this](CallNode *other){return after(*other);},others);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

class CallGraph
{
private:
  std::unordered_map<PetscInt,PetscInt> _idMap;
  std::vector<CallNode*> _graph;
  std::vector<CallNode*> _exec;
  PetscBool              _finalized = PETSC_FALSE;
  void                  *_userCtx   = nullptr;
  const std::string      _name;

  PetscErrorCode __topologicalSort(PetscInt v, std::vector<PetscBool> &visited)
  {
    const auto neigh = _graph[v]->inedges();

    PetscFunctionBegin;
    visited[v] = PETSC_TRUE;
    for (auto node = neigh.begin(); node != neigh.end(); ++node) {
      const auto imap = _idMap[(*node)->id()];

      if (!visited[imap]) {
        PetscErrorCode ierr;

        ierr = __topologicalSort(imap,visited);CHKERRQ(ierr);
      }
    }
    CHKERRCXX(_exec.push_back(_graph[v]));
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __printDeps(const CallNode &node)
  {
    PetscFunctionBegin;
    CHKERRCXX(std::cout<<"node "<<node.id()<<" [in: ");
    if (node.inedges().size()) {
      for (const auto &in : node.inedges()) {
        CHKERRCXX(std::cout<<in->id()<<",");
      }
    } else {
      CHKERRCXX(std::cout<<"none");
    }
    CHKERRCXX(std::cout<<" out: ");
    if (node.outedges().size()) {
      for (const auto &out : node.outedges()) {
        CHKERRCXX(std::cout<<out->id()<<",");
      }
    } else {
      CHKERRCXX(std::cout<<"none");
    }
    CHKERRCXX(std::cout<<" ]\n");
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __joinAncestors(CallNode &node)
  {
    const auto     &inedges  = node.inedges();
    PetscErrorCode  ierr;

    PetscFunctionBegin;
    switch (std::max(inedges.size()-node.outedges().size(),std::size_t(0))) {
    case 0:
      ierr = node.setStream(0);CHKERRQ(ierr);
      break;
    case 1:
      ierr = node.setStream(inedges[0]->stream());CHKERRQ(ierr);
    case 2:
      const auto min = std::min_element(inedges.begin(),inedges.end(),[](const CallNode* l, const CallNode* r) { return (l->id() < r->id());});
      const int minStream = (*min)->stream();
      auto destroyStream = [=](const CallNode *node)
      {
        if (node->stream() != minStream) {std::cout<<"destroying stream "<<node->stream()<<'\n';}
      };
      CHKERRCXX(std::for_each(inedges.begin(),inedges.end(),destroyStream));
      ierr = node.setStream(minStream);CHKERRQ(ierr);
      break;
    }
    PetscFunctionReturn(0);
  }

protected:
  // also used when graphs are composed, this allows the graph to act like a function
  // pointer
  PetscErrorCode operator()(void *ctx)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidPointer(ctx,1);
    ierr = finalize();CHKERRQ(ierr);
    std::cout<<"----- "<<_name<<" begin run\n";
    for (const auto *node : _exec) {
      const int incoming = node->inedges().size();
      const int outgoing = node->outedges().size();

      ierr = __printDeps(*node);CHKERRQ(ierr);
      std::cout<<"- node "<<node->id()<<" should join "<<incoming<<" and destroy "<<std::max(incoming-outgoing,0)<<" streams\n";
      ierr = __joinAncestors(const_cast<CallNode&>(*node));CHKERRQ(ierr);
      ierr = (*node)(_userCtx);CHKERRQ(ierr);
    }
    std::cout<<"----- "<<_name<<" end run\n";
    PetscFunctionReturn(0);
  }

public:
  CallGraph(const char name[] = "unnamed graph") : _name{name} { }

  ~CallGraph()
  {
    for (auto cnode : _graph) delete cnode;
    { std::cout<<_name<<" dtor"<<std::endl;}
  }

  const std::string& name() const { return _name;}

  CallNode* compose(CallGraph&);

  template <typename T>
  CallNode* emplace(T&&);

  template <typename... Argr, detail::enable_if_t<(sizeof...(Argr)>1)>* = nullptr>
  auto emplace(Argr&&... rest) -> decltype(std::make_tuple(emplace(std::forward<Argr>(rest))...));

  template <typename T, typename... Argr,
            detail::enable_if_t<detail::is_callable<T>::value && (sizeof...(Argr) > 0)>* = nullptr>
  CallNode* emplaceCall(T&&,Argr&&...);

  PetscErrorCode finalize();
  PetscErrorCode setUserContext(void*);
  PetscErrorCode run();
};

CallNode* CallGraph::compose(CallGraph &other)
{
  return emplace([&](void *ctx){return other(ctx);});
}

template <typename T>
CallNode* CallGraph::emplace(T &&ftor)
{
  static_assert(detail::is_callable<T>::value,"Entity passed to graph does not appear to be callable");
  _finalized = PETSC_FALSE;
  _graph.emplace_back(new CallNode{std::forward<T>(ftor)});
  _idMap[_graph.back()->id()] = _graph.size()-1;
  return _graph.back();
}

template <typename... Argr, detail::enable_if_t<(sizeof...(Argr) > 1)>*>
auto CallGraph::emplace(Argr&&... rest)
  -> decltype(std::make_tuple(emplace(std::forward<Argr>(rest))...))
{
  return std::make_tuple(emplace(std::forward<Argr>(rest))...);
}

template <typename T, typename... Argr,
          detail::enable_if_t<detail::is_callable<T>::value && (sizeof...(Argr) > 0)>*>
CallNode* CallGraph::emplaceCall(T &&f, Argr&&... args)
{
  _finalized = PETSC_FALSE;
  _graph.emplace_back(new CallNode{std::forward<T>(f),std::forward<Argr>(args)...});
  _idMap[_graph.back()->id()] = _graph.size()-1;
  return _graph.back();
}

PetscErrorCode CallGraph::finalize()
{
  PetscFunctionBegin;
  if (!_finalized) {
    std::vector<PetscBool> visited{_graph.size(),PETSC_FALSE};

    if (PetscUnlikelyDebug(!_userCtx)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_POINTER,"User must supply a user context before executing the graph");
    CHKERRCXX(_exec.clear());
    CHKERRCXX(_exec.reserve(_graph.size()));
    for (PetscInt i = 0; i < _graph.size(); ++i) {
      if (!visited[i]) {
        PetscErrorCode ierr;

        ierr = __topologicalSort(i,visited);CHKERRQ(ierr);
      }
    }
    _finalized = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CallGraph::setUserContext(void *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  _userCtx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode CallGraph::run()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*this)(_userCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCGRAPH_HPP */
