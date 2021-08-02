#ifndef PETSCGRAPHGENERAL_HPP
#define PETSCGRAPHGENERAL_HPP

#include <petsc/private/graphimpl.h>

#include <vector>
#include <map>
#include <stack>
#include <iostream>
#include <functional>

namespace Petsc {

namespace detail {

template <bool B, typename T = void>
#if __cplusplus >= 201402L
using enable_if_t = std::enable_if_t<B,T>;
#else
using enable_if_t = typename std::enable_if<B,T>::type;
#endif

template <typename T>
struct is_callable_class
{
private:
  typedef char (&yay)[1];
  typedef char (&nay)[2];

  struct Fallback
  {
    void operator()();
  };

  struct Derived : T, Fallback { };

  template <typename U, U> struct CheckCallable;

  template <typename  > static yay test(...);

  template <typename C> static nay test(CheckCallable<void(Fallback::*)(),&C::operator()>*);

public:
  static constexpr bool value = sizeof(test<Derived>(0)) == sizeof(yay);
};

template <typename T>
struct strip_function
{
  typedef typename std::remove_pointer<typename std::decay<T>::type>::type type;
};

template <typename T>
using strip_function_t = typename strip_function<T>::type;

template <typename T>
struct is_callable_function : std::is_function<strip_function_t<T>>
{ };

template <class T>
struct is_callable : std::conditional<
  std::is_class<T>::value,
  is_callable_class<T>,
  is_callable_function<T>
  >::type
{ };

template<bool...> struct bool_pack;

template<bool... bs>
//if any are false, they'll be shifted in the second version, so types won't match
using all_true_impl = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template <typename... Ts>
using all_true = all_true_impl<Ts::value...>;

template <typename T, typename... Ts>
using all_same = all_true<std::is_same<T,Ts>...>;

template<int ...> struct IntegerSequence { };

template<int N, int ...S> struct SequenceGenerator : SequenceGenerator<N-1, N-1, S...> { };

template<int ...S> struct SequenceGenerator<0, S...> : IntegerSequence<S...> { };

template <int I = 0, typename Tf, typename... Tp, detail::enable_if_t<I == sizeof...(Tp)>* = nullptr>
inline PetscErrorCode forEachInTuple(Tf &&fn, std::tuple<Tp...> &tuple) { return 0;}

template <int I = 0, typename Tf, typename... Tp, detail::enable_if_t<(I < sizeof...(Tp))>* = nullptr>
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
  struct CallableBase
  {
  public:
    using ptr_t = std::unique_ptr<const CallableBase>;

    virtual ~CallableBase()                        = default;
    virtual PetscErrorCode operator()(void*) const = 0;
    ptr_t clone() const { return ptr_t(cloneDerived());}

  protected:
    virtual CallableBase* cloneDerived() const = 0;
  };

  template <typename... Args>
  struct StaticCallable final : CallableBase
  {
  public:
    using signature_t = PetscErrorCode(void*,Args...);
    using function_t  = std::function<signature_t>;
    using argPack_t   = std::tuple<Args...>;
    using self_t = StaticCallable<Args...>;
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
    StaticCallable(T &&fn, argPack_t &&args)
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

protected:
  CallableBase::ptr_t   _functor;
  std::vector<PetscInt> _inedges;
  std::vector<PetscInt> _outedges;
  const PetscInt        _id;

  static PetscInt       counter;

public:
  // Default constructor
  CallNode() : _id{counter++} { std::cout<<"ctor "<<_id<<std::endl;}

  // Copy constructor
  CallNode(const CallNode &other)
    : _functor{other._functor->clone()}, _inedges{other._inedges}, _id{counter++}
  { }

  // Move constructor
  CallNode(CallNode &&other) noexcept = default;

  // Templated constructor with tuple
  template <typename T, typename ...Args>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor{new StaticCallable<Args...>{std::forward<T>(f),std::forward<std::tuple<Args...>>(args)}},
      _id{counter++}
  { std::cout<<"ctor "<<_id<<std::endl;}

  // Templated constructor with bare arguments
  template <typename T, typename ...Args>
  CallNode(T &&f, Args&&... args)
    : CallNode{std::forward<T>(f),std::make_tuple(std::forward<Args>(args)...)}
  { }

  // Destructor
  ~CallNode() {std::cout<<"dtor "<<_id<<std::endl;}

  // Copy assignment operator
  CallNode& operator=(const CallNode&);

  // Move assignment operator
  CallNode& operator=(CallNode &&) noexcept;

  // Call operator
  PetscErrorCode operator()(void*) const;

  // Private member accessors
  PetscInt id() const { return _id;}
  const std::vector<PetscInt>& inedges() const { return _inedges;}

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

class CallGraph
{
private:
  std::map<PetscInt,PetscInt> _idMap;
  std::vector<CallNode*>      _graph;
  std::vector<CallNode*>      _exec;
  PetscBool                   _finalized = PETSC_FALSE;
  void                       *_userCtx   = nullptr;

  PetscErrorCode __topologicalSort(PetscInt v, std::vector<PetscBool> &visited)
  {
    const auto neigh = _graph[v]->inedges();

    PetscFunctionBegin;
    visited[v] = PETSC_TRUE;
    for (auto i = neigh.begin(); i != neigh.end(); ++i) {
      const auto imap = _idMap[*i];

      if (!visited[imap]) {
        PetscErrorCode ierr;

        ierr = __topologicalSort(imap,visited);CHKERRQ(ierr);
      }
    }
    CHKERRCXX(_exec.push_back(_graph[v]));
    PetscFunctionReturn(0);
  }

public:
  ~CallGraph() { for (auto cnode : _graph) delete cnode; }

  template <typename T>
  CallNode* emplace(T&&);

  template <typename... Argr, detail::enable_if_t<(sizeof...(Argr)>1)>* = nullptr>
  auto emplace(Argr&&... rest) -> decltype(std::make_tuple(emplace(std::forward<Argr>(rest))...));

  template <typename T, typename... Argr,
            detail::enable_if_t<detail::is_callable<T>::value && (sizeof...(Argr) > 0)>* = nullptr>
  CallNode* emplaceCall(T&&,Argr&&...);

  PetscErrorCode finalize();
  PetscErrorCode setUserContext(void*);
  PetscErrorCode run(PetscBool);
};

CallNode& CallNode::operator=(const CallNode &other)
{
  PetscFunctionBegin;
  if (this != &other) {
    if (other._functor) {_functor = other._functor->clone();}
    _inedges = other._inedges;
  }
  PetscFunctionReturn(*this);
}

CallNode& CallNode::operator=(CallNode &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    _functor    = std::move(other._functor);
    _inedges = std::move(other._inedges);
  }
  PetscFunctionReturn(*this);
}

PetscErrorCode CallNode::operator()(void *ctx) const
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  std::cout<<"running operator "<<_id<<std::endl;
  CHKERRCXX(ierr = (*_functor)(ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CallNode::before(CallNode &other)
{
  PetscFunctionBegin;
  std::cout<<_id<<" before "<<other._id<<std::endl;
  CHKERRCXX(other._inedges.push_back(_id));
  CHKERRCXX(_outedges.push_back(other._id));
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
  CHKERRCXX(_inedges.push_back(other._id));
  CHKERRCXX(other._outedges.push_back(_id));
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
auto CallGraph::emplace(Argr&&... rest) -> decltype(std::make_tuple(emplace(std::forward<Argr>(rest))...))
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

PetscErrorCode CallGraph::run(PetscBool keep = PETSC_TRUE)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = finalize();CHKERRQ(ierr);
  for (const auto &node : _exec) {
    if (node->inedges().size()) {
      std::cout<<node->id()<<" has "<<node->inedges().size()<<" parent(s)"<<std::endl;
    }
    ierr = (*node)(_userCtx);CHKERRQ(ierr);
  }
  if (!keep) {
    _exec.clear();
    _finalized = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif
