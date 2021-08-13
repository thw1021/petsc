#ifndef PETSCGRAPH_HPP
#define PETSCGRAPH_HPP

#include <petsc/private/petscimpl.h>

#if defined(__cplusplus)

#include <vector>
#include <unordered_map>
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
  static PetscInt pool;
  void     *userCtx;
  PetscInt  stream;
  const     PetscBool enclosed;

  constexpr ExecutionContext(void *ctx, PetscInt strm) noexcept
    : userCtx(ctx),stream(strm),enclosed(strm >= 0 ? PETSC_TRUE : PETSC_FALSE) { }

  static PetscInt newStream() { return pool++;}
};
PetscInt ExecutionContext::pool = 0;

class CallNode
{
private:
  struct CallableBase // ABC for type-erasing the callable
  {
  public:
    using ptr_t = std::unique_ptr<const CallableBase>;

    virtual ~CallableBase() = default;
    virtual PetscErrorCode operator()(ExecutionContext*) const = 0;
    ptr_t clone() const { return ptr_t(__cloneDerived());}

  protected:
    virtual CallableBase* __cloneDerived() const = 0;
  };

  template <typename... Args>
  struct WrappedCallable final : CallableBase // actual callable
  {
  public:
    using signature_t = PetscErrorCode(ExecutionContext*,Args...);
    using function_t  = std::function<signature_t>;
    using argPack_t   = std::tuple<Args...>;
    using self_t      = WrappedCallable<Args...>;

  private:
    const function_t _functor;
    argPack_t        _params;

    self_t* __cloneDerived() const override final { return new self_t(*this);}

    template <int... S>
    PetscErrorCode __callFunc(ExecutionContext *ctx, detail::IntegerSequence<S...>) const
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = _functor(ctx,std::get<S>(_params)...);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

  public:
    template <typename T>
    WrappedCallable(T &&fn, argPack_t &&args)
      : _functor(std::forward<T>(fn)), _params(std::forward<argPack_t>(args))
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
    using self_t      = NativeCallable<Args...>;

  private:
    const function_t _functor;
    argPack_t        _params;

    self_t* __cloneDerived() const override final { return new self_t(*this);}

    template <int... S>
    PetscErrorCode __callFunc(detail::IntegerSequence<S...>) const
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = _functor(std::get<S>(_params)...);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

  public:
    template <typename T>
    NativeCallable(T &&fn, argPack_t &&args)
      : _functor(std::forward<T>(fn)), _params(std::forward<argPack_t>(args))
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

    constexpr Edge(PetscInt start, PetscInt end) : _start(start),_end(end) { }
  };

  using edge_t = std::shared_ptr<Edge>;

  CallableBase::ptr_t _functor = nullptr;
  std::vector<edge_t> _inedges;
  std::vector<edge_t> _outedges;
  std::string         _name;
  const PetscInt      _id;

  static PetscInt     counter;

  friend class CallGraph;

  PetscErrorCode __printDeps() const
  {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG)) {
      CHKERRCXX(std::cout<<"node "<<_id);
      if (!_name.empty()) {
        CHKERRCXX(std::cout<<" ("<<_name<<')');
      }
      CHKERRCXX(std::cout<<" [in: ");
      if (_inedges.empty()) {
        CHKERRCXX(std::cout<<"none");
      } else {
        for (const auto &in : _inedges) {
          CHKERRCXX(std::cout<<in->_start<<",");
        }
      }
      CHKERRCXX(std::cout<<" out: ");
      if (_outedges.empty()) {
        CHKERRCXX(std::cout<<"none");
      } else {
        for (const auto &out : _outedges) {
          CHKERRCXX(std::cout<<out->_end<<",");
        }
      }
      CHKERRCXX(std::cout<<" ]"<<std::endl);
    }
    PetscFunctionReturn(0);
  }

public:
  // Default constructor
  CallNode() : _id(counter++) { }

  // Named Default constructor
  explicit CallNode(const char name[]) : _name(name),_id(counter++) { }

  // Copy constructor
  CallNode(const CallNode &other)
    : _functor(other._functor->clone()), _inedges(other._inedges), _outedges(other._outedges),
      _id(counter++)
  { }

  // Move constructor
  CallNode(CallNode &&other) noexcept = default;

  // Templated constructor with required tuple of arguments, only enabled if first
  // argument is NOT a void *
  template <typename T, typename ...Args,
            detail::enable_if_t<
              !std::is_same<
                typename detail::function_traits<T>::template arg<0>::type,
                ExecutionContext*
                >::value
              >* = nullptr>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor(new NativeCallable<Args...>(std::forward<T>(f),std::forward<std::tuple<Args...>>(args))),
      _id(counter++)
  { static_assert(sizeof...(Args), "need args for native call");}

  // Templated constructor with optional tuple of arguments, only enabled if first
  // argument is a void *
  template <typename T, typename ...Args,
            detail::enable_if_t<
              std::is_same<
                typename detail::function_traits<T>::template arg<0>::type,
                ExecutionContext*
                >::value
              >* = nullptr>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor(new WrappedCallable<Args...>(std::forward<T>(f),std::forward<std::tuple<Args...>>(args))),
      _id(counter++)
  { }

  // Templated constructor with bare arguments
  template <typename T, typename ...Args>
  CallNode(T &&f, Args&&... args)
    : CallNode(std::forward<T>(f),std::make_tuple(std::forward<Args>(args)...))
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
  if (_functor) {
    std::cout<<"- running on stream "<<ctx->stream<<":\n";
    ierr = (*_functor)(ctx);CHKERRQ(ierr);
  }
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
  CallNode    _begin;
  CallNode    _end;
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

  // this routine serves 2 purposes:
  // 1. To find and remove the fake dependencies on the _begin and _end nodes. Since these
  //    are shared_ptr's we need to delete them in both _begin and _end and the nodes they
  //    are attached to in _graph.
  // 2. To reset and clear _exec
  PetscErrorCode __resetClosure(CallNode *closureBegin, CallNode *closureEnd)
  {
    // normally when you erase an element in a vector all the elements after it must be
    // copied over by one to plug the gap. this routine optimizes this by copying the last
    // element of the vector in place of the element to be deleted, and then deleting the
    // element at the back.
    const auto quickDelete = [](const CallNode::edge_t &edge, std::vector<CallNode::edge_t> &vec)
    {
      PetscFunctionBegin;
      auto loc = std::find(vec.begin(),vec.end(),edge);
      *loc = std::move(vec.back());
      CHKERRCXX(vec.pop_back());
      PetscFunctionReturn(0);
    };
    PetscErrorCode ierr;

    PetscFunctionBegin;
    for (const auto &edge : closureBegin->_outedges) {
      ierr = quickDelete(edge,_graph[_idMap[edge->_end]]->_inedges);CHKERRQ(ierr);
    }
    CHKERRCXX(closureBegin->_outedges.clear());
    for (const auto &edge : closureEnd->_inedges) {
      ierr = quickDelete(edge,_graph[_idMap[edge->_start]]->_outedges);CHKERRQ(ierr);
    }
    CHKERRCXX(closureEnd->_inedges.clear());
    PetscFunctionReturn(0);
  }

  PetscErrorCode __finalize(CallNode *closureBegin, CallNode *closureEnd)
  {
    PetscFunctionBegin;
    if (!_finalized) {
      std::vector<PetscBool> visited(_graph.size(),PETSC_FALSE);
      PetscErrorCode         ierr;

      PetscValidPointer(closureBegin,1);
      PetscValidPointer(closureEnd,2);
      if (PetscUnlikelyDebug(!_userCtx)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_POINTER,"User must supply a user context before executing the graph");
      // reset our executable graph
      CHKERRCXX(_exec.clear());
      // reserve enough for the graph+enclosures
      CHKERRCXX(_exec.reserve(_graph.size()+2));
      ierr = __resetClosure(closureBegin,closureEnd);CHKERRQ(ierr);
      // install the start of the closure
      CHKERRCXX(_exec.push_back(closureBegin));
      for (PetscInt i = 0; i < _graph.size(); ++i) {
        if (!visited[i]) {ierr = __topologicalSort(i,visited);CHKERRQ(ierr);}
        // check if the node has no inedges, if not it is a "start" node
        if (_graph[i]->_inedges.empty()) {_graph[i]->after(*closureBegin);}
        // check if the node has no outedges, if not is is an "end" node
        if (_graph[i]->_outedges.empty()) {_graph[i]->before(*closureEnd);}
      }
      ierr = closureBegin->__printDeps();CHKERRQ(ierr);
      ierr = closureEnd->__printDeps();CHKERRQ(ierr);
      // finish the closure
      CHKERRCXX(_exec.push_back(closureEnd));
      _finalized = PETSC_TRUE;
    }
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __joinAncestors(const CallNode &node, ExecutionContext &ctx)
  {
    PetscFunctionBegin;
    switch (node._inedges.size()) {
    case 0:
      // this node has no ancestors and is therefore *a* starting node of the graph
      std::cout<<"no ancestors\n";
      // however if the ctx is enclosed then it should not create a new stream, as the
      // enclosing scope has already set it
      if (!ctx.enclosed) ctx.stream = ctx.newStream();
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
        std::cout<<"-- joining ancestor stream "<<edge->_stream<<'\n';
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
      // we have no decendants, meaning we should destroy this stream (unless we are in
      // someone elses closure)
      if (!ctx.enclosed) std::cout<<"-- destroying stream "<<ctx.stream<<'\n';
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

  // when composing graphs we need to be able to distinguish the hosted graph from the
  // hosting graph (since 1 of the graphs needs to manage the others streams and
  // dependencies). Hence we have both "run" and operator(). run is only every called from
  // the hosting graph, whereas the hosted graph is invoked via operator().
  PetscErrorCode operator()(ExecutionContext *ctx, CallNode *closureBegin, CallNode *closureEnd)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidPointer(ctx,1);
    ierr = __finalize(closureBegin,closureEnd);CHKERRQ(ierr);
    std::cout<<"----- "<<_name<<" begin run\n";
    for (const auto &node : _exec) {
      ierr = node->__printDeps();CHKERRQ(ierr);
      ierr = __joinAncestors(*node,*ctx);CHKERRQ(ierr);
      ierr = node->run(ctx);CHKERRQ(ierr);
      ierr = __forkDecendants(*node,*ctx);CHKERRQ(ierr);
    }
    std::cout<<"----- "<<_name<<" end run\n";
    PetscFunctionReturn(0);
  }

public:
  CallGraph(const char name[] = "anonymous graph")
    : _name(name),_begin((_name+" closure begin").c_str()),_end((_name+" closure end").c_str()) { }

  ~CallGraph()
  {
    for (auto cnode : _graph) delete cnode;
    std::cout<<_name<<" dtor\n";
  }

  PetscErrorCode setName(const char*);
  const std::string& name() const { return _name;}

  PetscErrorCode compose(CallGraph&,CallNode*&);

  template <typename T>
  CallNode* emplace(T&&);

  template <typename... Argr, detail::enable_if_t<(sizeof...(Argr)>1)>* = nullptr>
  auto emplace(Argr&&... rest) -> decltype(std::make_tuple(emplace(std::forward<Argr>(rest))...));

  template <typename T, typename... Argr,
            detail::enable_if_t<detail::is_callable<T>::value && (sizeof...(Argr) > 0)>* = nullptr>
  CallNode* emplaceCall(T&&,Argr&&...);

  PetscErrorCode setUserContext(void*);
  PetscErrorCode run(PetscInt = -1);
};

PetscErrorCode CallGraph::compose(CallGraph &other, CallNode *&node)
{
  PetscFunctionBegin;
  node = emplace([&](ExecutionContext *ctx)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = other.run(ctx->stream);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  });
  PetscFunctionReturn(0);

}

template <typename T>
CallNode* CallGraph::emplace(T &&ftor)
{
  static_assert(detail::is_callable<T>::value,"Entity passed to graph does not appear to be callable");
  _finalized = PETSC_FALSE;
  _graph.emplace_back(new CallNode(std::forward<T>(ftor)));
  _idMap[_graph.back()->_id] = _graph.size()-1;
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
  _graph.emplace_back(new CallNode(std::forward<T>(f),std::forward<Argr>(args)...));
  _idMap[_graph.back()->_id] = _graph.size()-1;
  return _graph.back();
}

PetscErrorCode CallGraph::setUserContext(void *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  _userCtx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode CallGraph::run(PetscInt stream)
{
  ExecutionContext ctx(_userCtx,stream);
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = (*this)(&ctx,&_begin,&_end);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCGRAPH_HPP */
