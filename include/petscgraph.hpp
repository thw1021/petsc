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

struct ExecutionContext
{
  void     *userCtx;
  PetscInt  stream;

  constexpr ExecutionContext(void *ctx, PetscInt strm) noexcept : userCtx{ctx},stream{strm} { }

  static PetscInt newStream()
  {
    static PetscInt pool = 0;
    return pool++;
  }
};

class CallNode
{
private:
  struct CallableBase // ABC for type-erasing the callable
  {
  public:
    using ptr_t = std::unique_ptr<const CallableBase>;

    virtual ~CallableBase() = default;
    virtual PetscErrorCode operator()(ExecutionContext*) const = 0;
    ptr_t clone() const { return ptr_t{cloneDerived()};}

  protected:
    virtual CallableBase* cloneDerived() const = 0;
  };

  template <typename... Args>
  struct WrappedCallable final : CallableBase // actual callable
  {
  public:
    using signature_t = PetscErrorCode(ExecutionContext*,Args...);
    using function_t  = std::function<signature_t>;
    using argPack_t   = std::tuple<Args...>;
    using self_t = WrappedCallable<Args...>;
    using base_t = CallableBase;

  private:
    const function_t _functor;
    argPack_t        _params;

    template <int... S>
    PetscErrorCode __callFunc(ExecutionContext *ctx, detail::IntegerSequence<S...>) const
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

    PetscErrorCode operator()(ExecutionContext *ctx) const override final
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

    PetscErrorCode operator()(ExecutionContext *ctx) const override final
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      (void)(ctx); // unused
      ierr = __callFunc(detail::SequenceGenerator<sizeof...(Args)>());CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  };

  struct Edge
  {
    const PetscInt _start,_end;
    PetscInt       _stream = 0;

    constexpr Edge(PetscInt start, PetscInt end) : _start{start},_end{end} { }
  };

protected:
  friend class CallGraph;

  CallableBase::ptr_t    _functor = nullptr;
  std::vector<std::shared_ptr<Edge>> _inedges;
  std::vector<std::shared_ptr<Edge>> _outedges;
  mutable PetscInt       _stream = 0;
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
                ExecutionContext*
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
                ExecutionContext*
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

  // Private member accessors
  PetscInt id() const { return _id;}

  // Order enforcement
  PetscErrorCode before(CallNode&);
  template <typename... Args,
            detail::enable_if_t<detail::all_same<CallNode*,Args...>::value>* = nullptr>
  PetscErrorCode before(std::tuple<Args...>&);
  PetscErrorCode after(CallNode&);
  template <typename... Args,
            detail::enable_if_t<detail::all_same<CallNode*,Args...>::value>* = nullptr>
  PetscErrorCode after(std::tuple<Args...>&);

  // Execution
  PetscErrorCode run(ExecutionContext*) const;
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

PetscErrorCode CallNode::before(CallNode &other)
{
  PetscFunctionBegin;
  {
    std::cout<<_id<<" before "<<other._id<<std::endl;
    CHKERRCXX(_outedges.push_back(std::make_shared<Edge>(_id,other._id)));
    CHKERRCXX(other._inedges.push_back(_outedges.back()));
  }
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
  CHKERRCXX(_inedges.push_back(std::make_shared<Edge>(other._id,_id)));
  CHKERRCXX(other._outedges.push_back(_inedges.back()));
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

PetscErrorCode CallNode::run(ExecutionContext *ctx) const
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  std::cout<<"- running on stream "<<ctx->stream<<":\n";
  ierr = (*_functor)(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

class CallGraph
{
private:
  using map_t   = std::unordered_map<PetscInt,PetscInt>;
  using graph_t = std::vector<CallNode*>;

  std::string _name;
  map_t       _idMap;
  graph_t     _graph;
  graph_t     _exec;
  PetscBool   _finalized = PETSC_FALSE;
  void       *_userCtx   = nullptr;

  PetscErrorCode __topologicalSort(PetscInt v, std::vector<PetscBool> &visited)
  {
    PetscFunctionBegin;
    visited[v] = PETSC_TRUE;
    for (const auto &edge : _graph[v]->_inedges) {
      const auto imap = _idMap[edge->_start];

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
    CHKERRCXX(std::cout<<"node "<<node._id<<" [in: ");
    if (node._inedges.size()) {
      for (const auto &in : node._inedges) {
        CHKERRCXX(std::cout<<in->_start<<",");
      }
    } else {
      CHKERRCXX(std::cout<<"none");
    }
    CHKERRCXX(std::cout<<" out: ");
    if (node._outedges.size()) {
      for (const auto &out : node._outedges) {
        CHKERRCXX(std::cout<<out->_end<<",");
      }
    } else {
      CHKERRCXX(std::cout<<"none");
    }
    CHKERRCXX(std::cout<<" ]\n");
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __joinAncestors(const CallNode &node, ExecutionContext &ctx)
  {
    PetscFunctionBegin;
    switch (node._inedges.size()) {
    case 0:
      // this node has no ancestors and is therefore *a* starting node of the graph
      ctx.stream = ctx.newStream();
      break;
    default:
      // we have ancestors, meaning we must pick one of their streams and join all others
      // on that stream. we arbitrarily choose the first ancestor as the stream donor.
      ctx.stream = node._inedges[0]->_stream;
      // destroy all other streams except for the one we want to use. In the case we have
      // only 1 ancestor, this loop never fires
      CHKERRCXX(std::for_each(node._inedges.begin()+1,node._inedges.end(),
      [](const std::shared_ptr<CallNode::Edge> &edge)
      {
        std::cout<<"-- destroying ancestor stream "<<edge->_stream<<'\n';
      }));
      break;
    }
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __forkDecendants(const CallNode &node, const ExecutionContext &ctx)
  {
    PetscFunctionBegin;
    switch (node._outedges.size()) {
    case 0:
      // we have no decendants, meaning we should destroy this stream
      std::cout<<"-- destroying stream "<<ctx.stream<<'\n';
      break;
    default:
      // we have decendants, so we need to fork some streams off of the current one,
      // although we need only create n-1 streams since we can give ours away. Once again
      // arbitrarily pick our first outbound edge as the recipient of our stream
      node._outedges[0]->_stream = ctx.stream;
      // again, for more than 1 decendant this loop never fires
      CHKERRCXX(std::for_each(node._outedges.begin()+1,node._outedges.end(),
      [&](const std::shared_ptr<CallNode::Edge> &edge)
      {
        edge->_stream = ctx.newStream();
        std::cout<<"-- creating decendant stream "<<edge->_stream<<'\n';
      }));
      break;
    }
    PetscFunctionReturn(0);
  }

protected:
  // when composing graphs we need to be able to distinguish the hosted graph from the
  // hosting graph (since 1 of the graphs needs to manage the others streams and
  // dependencies). Hence we have both "run" and operator(). run is only every called from
  // the hosting graph, whereas the hosted graph is invoked via operator().
  PetscErrorCode operator()(ExecutionContext *ctx)
  {
    const void     *oldCtx = _userCtx;
    PetscErrorCode  ierr;

    PetscFunctionBegin;
    PetscValidPointer(ctx,1);
    _userCtx = ctx->userCtx;
    ierr = finalize();CHKERRQ(ierr);
    std::cout<<"----- "<<_name<<" begin run\n";
    for (const auto *const node : _exec) {
      ierr = __printDeps(*node);CHKERRQ(ierr);
      ierr = __joinAncestors(*node,*ctx);CHKERRQ(ierr);
      ierr = node->run(ctx);CHKERRQ(ierr);
      ierr = __forkDecendants(*node,*ctx);CHKERRQ(ierr);
    }
    _userCtx = const_cast<void*>(oldCtx);
    std::cout<<"----- "<<_name<<" end run\n";
    PetscFunctionReturn(0);
  }

public:
  CallGraph(const char name[] = "unnamed graph") : _name{name} { }

  ~CallGraph()
  {
    for (auto cnode : _graph) delete cnode;
    std::cout<<_name<<" dtor\n";
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
  return emplace([&](ExecutionContext *ctx){return other(ctx);});
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
  ExecutionContext ctx{_userCtx,0};
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = (*this)(&ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCGRAPH_HPP */
