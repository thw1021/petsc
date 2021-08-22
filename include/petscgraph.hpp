#ifndef PETSCGRAPH_H
#define PETSCGRAPH_H

#include <petscsys.h>
#include <petscgraphtypes.h>

PETSC_EXTERN PetscErrorCode PetscCallGraphCreate(MPI_Comm,PetscCallGraph*);

#if defined(__cplusplus)

#include <vector>
#include <array>
#include <tuple>
#include <unordered_map>
#include <algorithm>
#include <type_traits>
#include <iostream>
#include <functional>

namespace Petsc {

#if __cplusplus >= 201402L // (c++14)
using std::enable_if_t;
using std::decay_t;
using std::remove_pointer_t;
using std::remove_reference_t;
using std::conditional_t;
using std::tuple_element_t;
using std::index_sequence;
using std::make_index_sequence;
#else
template <bool B, typename T = void>
using enable_if_t        = typename std::enable_if<B,T>::type;
template <class T>
using decay_t            = typename std::decay<T>::type;
template <class T>
using remove_pointer_t   = typename std::remove_pointer<T>::type;
template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;
template< bool B, class T, class F >
using conditional_t      = typename std::conditional<B,T,F>::type;
template <std::size_t I, class T>
using tuple_element_t    = typename std::tuple_element<I,T>::type;

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

  template <typename C> static yay test(decltype(&C::operator()));
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

template <typename T, typename... Ts>
using all_same = detail::all_true<std::is_same<T,Ts>::value...>;

template <typename T>
struct is_callable : conditional_t<
  std::is_class<T>::value,
  detail::is_callable_class<T>,
  detail::is_callable_function<T>
  >
{ };

template <typename T> struct function_traits;

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

struct ExecutionContext
{
  void     *userCtx;
  PetscInt  stream;
  PetscBool repeatCall = PETSC_FALSE;
  const     PetscBool enclosed;

  constexpr ExecutionContext(void *ctx, PetscInt strm) noexcept
    : userCtx(ctx),stream(strm),enclosed(strm >= 0 ? PETSC_TRUE : PETSC_FALSE) { }

  PetscErrorCode repeat(PetscBool rep = PETSC_TRUE)
  {
    PetscFunctionBegin;
    repeatCall = rep;
    PetscFunctionReturn(0);
  }

  static PetscInt newStream()
  {
    static PetscInt pool = 0;
    return pool++;
  }

  PetscErrorCode getUserContext(void **ctx) const
  {
    PetscFunctionBegin;
    *ctx = userCtx;
    PetscFunctionReturn(0);
  }
};

class CallGraph;

class CallNode
{
public:
  enum class NodeKind : int {
    EMPTY,
    CALLABLE,
    COMPOSITION
  };

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
    const argPack_t  _params;

    self_t* __cloneDerived() const override final { return new self_t(*this);}

    template <std::size_t... S>
    PetscErrorCode __callFunc(ExecutionContext *ctx, index_sequence<S...>) const
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
      ierr = __callFunc(ctx,make_index_sequence<sizeof...(Args)>());CHKERRQ(ierr);
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
    const argPack_t  _params;

    self_t* __cloneDerived() const override final { return new self_t(*this);}

    template <std::size_t... S>
    PetscErrorCode __callFunc(index_sequence<S...>) const
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

    PetscErrorCode operator()(PETSC_UNUSED ExecutionContext *ctx) const override final
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = __callFunc(make_index_sequence<sizeof...(Args)>());CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  };

  struct Edge
  {
    const PetscInt _start,_end;
    PetscInt       _stream = 0;

    constexpr Edge(PetscInt start, PetscInt end) noexcept : _start(start),_end(end) { }
  };

  // if you can figure out how to not get linker errors when using a static data member in
  // the ctors then go ahead and nix this useless function
  static PetscInt __counter()
  {
    static PetscInt counter = 0;
    return counter++;
  }

  using edge_t = std::shared_ptr<Edge>;

  CallableBase::ptr_t _functor = nullptr;
  std::vector<edge_t> _inedges;
  std::vector<edge_t> _outedges;
  std::string         _name = "anonymous node";
  NodeKind            _kind = NodeKind::EMPTY;
  const PetscInt      _id   = __counter();// = counter++;

  friend class CallGraph;

  PetscErrorCode __printDeps() const
  {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG)) {
      CHKERRCXX(std::cout<<"node "<<_id);
      if (!_name.empty()) CHKERRCXX(std::cout<<" ("<<_name<<')');
      CHKERRCXX(std::cout<<" [in: ");
      if (_inedges.empty()) {
        CHKERRCXX(std::cout<<"none");
      } else {
        for (const auto &in : _inedges) CHKERRCXX(std::cout<<in->_start<<",");
      }
      CHKERRCXX(std::cout<<" out: ");
      if (_outedges.empty()) {
        CHKERRCXX(std::cout<<"none");
      } else {
        for (const auto &out : _outedges) CHKERRCXX(std::cout<<out->_end<<",");
      }
      CHKERRCXX(std::cout<<" ]"<<std::endl);
    }
    PetscFunctionReturn(0);
  }

public:
  // Default constructor
  CallNode() { }

  // Named Default constructor
  CallNode(std::string &&name) : _name(std::forward<std::string>(name)) { }
  CallNode(const std::string &name) : _name(name) { }
  CallNode(const char name[]) : _name(name) { }

  // Copy constructor
  CallNode(const CallNode &other)
    : _functor(other._functor->clone()), _inedges(other._inedges), _outedges(other._outedges),
      _name(other._name), _kind(other._kind)
  { }

  // Move constructor
  CallNode(CallNode &&other) noexcept = default;

  // Composed constructor defined out of line as it needs CallGraph definition
  CallNode(CallGraph &graph);

  // Templated constructor with required tuple of arguments, only enabled if first
  // argument is NOT an ExecutionContext*
  template <typename T, typename ...Args,
            enable_if_t<
              !std::is_same<
                typename function_traits<T>::template argument<0>::type,
                ExecutionContext*
                >::value
              >* = nullptr>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor(new NativeCallable<Args...>(std::forward<T>(f),std::forward<std::tuple<Args...>>(args))),
      _kind(NodeKind::CALLABLE)
  {
    static_assert(sizeof...(Args),"need args for native call");
    std::cout<<"native ctor\n";
  }

  // Templated constructor with optional tuple of arguments, only enabled if first
  // argument is an ExecutionContext*
  template <typename T, typename ...Args,
            enable_if_t<
              std::is_same<
                typename function_traits<T>::template argument<0>::type,
                ExecutionContext*
                >::value
              >* = nullptr>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : _functor(new WrappedCallable<Args...>(std::forward<T>(f),std::forward<std::tuple<Args...>>(args))),
      _kind(NodeKind::CALLABLE)
  { std::cout<<"wrapped ctor\n"; }

  // Templated constructor with bare arguments
  template <typename T, typename ...Args>
  CallNode(T &&f, Args&&... args)
    : CallNode(std::forward<T>(f),std::make_tuple(std::forward<Args>(args)...))
  { }

  // Destructor
  ~CallNode() { std::cout<<"node dtor "<<_id<<" ("<<_name<<")\n"; }

  // Copy assignment operator
  CallNode& operator=(const CallNode&);

  // Move assignment operator
  CallNode& operator=(CallNode &&) noexcept;

  // Private member accessors
  NodeKind kind() const { return _kind; }
  const std::string kind_to_str() const
  {
    switch(_kind) {
    case NodeKind::EMPTY:       return "EMPTY";
    case NodeKind::CALLABLE:    return "CALLABLE";
    case NodeKind::COMPOSITION: return "COMPOSITION";
    default:                    return "";
    }
  }
  PetscInt id() const { return _id; }
  const std::string& name() const { return _name; }
  PetscErrorCode setName(const std::string &name)
  {
    PetscFunctionBegin;
    _name = name;
    PetscFunctionReturn(0);
  }

  // Order enforcement
  PetscErrorCode before(CallNode*);
  template <std::size_t N>
  PetscErrorCode before(std::array<CallNode*,N>&);
  PetscErrorCode after(CallNode*);
  template <std::size_t N>
  PetscErrorCode after(std::array<CallNode*,N>&);

  // Execution
  PetscErrorCode run(ExecutionContext*) const;
};

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
      if (!visited[_idMap[edge->_start]]) {
        PetscErrorCode ierr;

        ierr = __topologicalSort(_idMap[edge->_start],visited);CHKERRQ(ierr);
      }
    }
    CHKERRCXX(_exec.push_back(_graph[v]));
    PetscFunctionReturn(0);
  }

  // this routine finds and removes the fake dependencies on the _begin and _end
  // nodes. Since these are shared_ptr's we need to delete them in both _begin and _end
  // and the nodes they are attached to in _graph.
  PetscErrorCode __resetClosure()
  {
    PetscFunctionBegin;
    for (const auto &edge : _begin._outedges) {
      auto &vec = _graph[_idMap[edge->_end]]->_inedges;

      CHKERRCXX(vec.erase(std::remove_if(vec.begin(),vec.end(),
      [&](const CallNode::edge_t &found)
      {
        return edge == found;
      })));
    }
    CHKERRCXX(_begin._outedges.clear());
    for (const auto &edge : _end._inedges) {
      auto &vec = _graph[_idMap[edge->_start]]->_outedges;

      CHKERRCXX(vec.erase(std::remove_if(vec.begin(),vec.end(),
      [&](const CallNode::edge_t &found)
      {
        return edge == found;
      })));
    }
    CHKERRCXX(_end._inedges.clear());
    PetscFunctionReturn(0);
  }

  PetscErrorCode __finalize()
  {
    PetscFunctionBegin;
    if (!_finalized) {
      const PetscInt         graphSize = _graph.size();
      std::vector<PetscBool> visited(graphSize,PETSC_FALSE);
      PetscErrorCode         ierr;

      // reset our executable graph
      CHKERRCXX(_exec.clear());
      // reserve enough for the graph+closure nodes
      CHKERRCXX(_exec.reserve(graphSize+2));
      ierr = __resetClosure();CHKERRQ(ierr);
      // install the start of the closure
      CHKERRCXX(_exec.push_back(&_begin));
      for (PetscInt i = 0; i < graphSize; ++i) {
        std::cout<<_graph[i]->kind_to_str()<<'\n';
        if (!visited[i]) {ierr = __topologicalSort(i,visited);CHKERRQ(ierr);}
        // if the node has no inedges it is a "start" node
        if (_graph[i]->_inedges.empty()) {ierr = _graph[i]->after(&_begin);CHKERRQ(ierr);}
        // if the node has no outedges it is an "end" node
        if (_graph[i]->_outedges.empty()) {ierr = _graph[i]->before(&_end);CHKERRQ(ierr);}
      }
      ierr = _begin.__printDeps();CHKERRQ(ierr);
      ierr = _end.__printDeps();CHKERRQ(ierr);
      // finish the closure
      CHKERRCXX(_exec.push_back(&_end));
      _finalized = PETSC_TRUE;
    }
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __joinAncestors(const CallNode &node, ExecutionContext &ctx)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = node.__printDeps();CHKERRQ(ierr);
    if (node._inedges.empty()) {
      // this node has no ancestors and is therefore *a* starting node of the graph
      std::cout<<"no ancestors\n";
      // however if the ctx is enclosed then it should not create a new stream, as the
      // enclosing scope has already set it
      if (!ctx.enclosed) ctx.stream = ctx.newStream();
    } else {
      // we have ancestors, meaning we must pick one of their streams and join all others
      // on that stream. we arbitrarily choose the first ancestor as the stream donor.
      ctx.stream = node._inedges[0]->_stream;
      // destroy all other streams except for the one we want to use. In the case we have
      // only 1 ancestor, this loop never fires
      CHKERRCXX(std::for_each(std::next(node._inedges.cbegin()),node._inedges.cend(),
      [](const CallNode::edge_t &edge)
      {
        std::cout<<"-- joining ancestor stream "<<edge->_stream<<'\n';
      }));
    }
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __forkDecendants(const CallNode &node, const ExecutionContext &ctx)
  {
    PetscFunctionBegin;
    if (node._outedges.empty()) {
      // we have no decendants, meaning we should destroy this stream (unless we are in
      // someone elses closure)
      if (!ctx.enclosed) std::cout<<"-- destroying stream "<<ctx.stream<<'\n';
    } else {
      // we have decendants, so we need to fork some streams off of the current one,
      // although we need only create n-1 streams since we can give ours away. Once again
      // arbitrarily pick our first outbound edge as the recipient of our stream
      node._outedges[0]->_stream = ctx.stream;
      // again, for more than 1 decendant this loop never fires
      CHKERRCXX(std::for_each(std::next(node._outedges.cbegin()),node._outedges.cend(),
      [&](const CallNode::edge_t &edge)
      {
        edge->_stream = ctx.newStream();
        std::cout<<"-- creating decendant stream "<<edge->_stream<<'\n';
      }));
    }
    PetscFunctionReturn(0);
  }

public:
  CallGraph(const char name[] = "anonymous graph")
    : _name(name),_begin(_name+" closure begin"),_end(_name+" closure end") { }

  ~CallGraph()
  {
    for (auto cnode : _graph) delete cnode;
    std::cout<<_name<<" dtor\n";
  }

  const std::string& name() const { return _name;}
  PetscErrorCode setName(const std::string &name)
  {
    PetscFunctionBegin;
    _name = name;
    PetscFunctionReturn(0);
  }

  template <typename T>
  CallNode* emplace(T&&);

  template <typename... Args, enable_if_t<(sizeof...(Args)>1)>* = nullptr>
  std::array<CallNode*,sizeof...(Args)> emplace(Args&&...);

  template <typename T, typename... Argr,
            enable_if_t<is_callable<T>::value && (sizeof...(Argr) > 0)>* = nullptr>
  CallNode* emplaceCall(T&&,Argr&&...);

  PetscErrorCode setUserContext(void*);
  PetscErrorCode run(PetscInt = -1);
};

inline CallNode::CallNode(CallGraph &graph)
  : _functor(new WrappedCallable<>([&](ExecutionContext *ctx)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = graph.run(ctx->stream);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  },std::make_tuple())), _kind(NodeKind::COMPOSITION)
{ std::cout<<"COMPOSITION CTOR\n";}

template <std::size_t N>
inline PetscErrorCode CallNode::before(std::array<CallNode*,N> &others)
{
  PetscFunctionBegin;
  for (auto &other : others) {
    PetscErrorCode ierr;

    ierr = before(other);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <std::size_t N>
inline PetscErrorCode CallNode::after(std::array<CallNode*,N> &others)
{
  PetscFunctionBegin;
  for (auto &other : others) {
    PetscErrorCode ierr;

    ierr = after(other);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::before(CallNode *other)
{
  PetscFunctionBegin;
  std::cout<<_id<<" ("<<_name<<") before "<<other->_id<<" ("<<other->_name<<")\n";
  CHKERRCXX(_outedges.push_back(std::make_shared<Edge>(_id,other->_id)));
  CHKERRCXX(other->_inedges.push_back(_outedges.back()));
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::after(CallNode *other)
{
  PetscFunctionBegin;
  std::cout<<_id<<" ("<<_name<<") after "<<other->_id<<" ("<<other->_name<<")\n";
  CHKERRCXX(_inedges.push_back(std::make_shared<Edge>(other->_id,_id)));
  CHKERRCXX(other->_outedges.push_back(_inedges.back()));
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::run(ExecutionContext *ctx) const
{
  PetscFunctionBegin;
  if (_kind != NodeKind::EMPTY) {
    PetscErrorCode ierr;

    std::cout<<"- running on stream "<<ctx->stream<<":\n";
    ierr = (*_functor)(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

inline CallNode& CallNode::operator=(const CallNode &other)
{
  PetscFunctionBegin;
  if (this != &other) {
    if (other._functor) _functor = other._functor->clone();
    _inedges  = other._inedges;
    _outedges = other._outedges;
  }
  PetscFunctionReturn(*this);
}

inline CallNode& CallNode::operator=(CallNode &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    _functor  = std::move(other._functor);
    _inedges  = std::move(other._inedges);
    _outedges = std::move(other._outedges);
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline CallNode* CallGraph::emplace(T &&fn)
{
  PetscFunctionBegin;
  // we allow CallGraph through this check as it uses CallGraph::run() as the canonical
  // operator(). We could instead explicitly overload this routine for CallGraph&, but
  // that is a lot code bloat for just one line
  static_assert(std::is_same<remove_reference_t<T>,CallGraph>::value || is_callable<T>::value,
                "Entity passed to graph does not appear to be callable");
  _finalized = PETSC_FALSE;
  _graph.emplace_back(new CallNode(std::forward<T>(fn)));
  _idMap[_graph.back()->_id] = _graph.size()-1;
  PetscFunctionReturn(_graph.back());
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 1)>*>
inline std::array<CallNode*,sizeof...(Args)> CallGraph::emplace(Args&&... rest)
{
  _graph.reserve(_graph.size()+sizeof...(Args));
  return {emplace(std::forward<Args>(rest))...};
}

template <typename T, typename... Argr,
          enable_if_t<is_callable<T>::value && (sizeof...(Argr) > 0)>*>
inline CallNode* CallGraph::emplaceCall(T &&f, Argr&&... args)
{
  PetscFunctionBegin;
  _finalized = PETSC_FALSE;
  _graph.emplace_back(new CallNode(std::forward<T>(f),std::forward<Argr>(args)...));
  _idMap[_graph.back()->_id] = _graph.size()-1;
  PetscFunctionReturn(_graph.back());
}

inline PetscErrorCode CallGraph::setUserContext(void *ctx)
{
  PetscFunctionBegin;
  _userCtx = ctx;
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallGraph::run(PetscInt stream)
{
  ExecutionContext ctx(_userCtx,stream);
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = __finalize();CHKERRQ(ierr);
  std::cout<<"----- "<<_name<<" begin run\n";
  for (const auto &node : _exec) {
    ierr = __joinAncestors(*node,ctx);CHKERRQ(ierr);
    do {ierr = node->run(&ctx);CHKERRQ(ierr);} while (ctx.repeatCall);
    ierr = __forkDecendants(*node,ctx);CHKERRQ(ierr);
  }
  std::cout<<"----- "<<_name<<" end run\n";
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCGRAPH_H */
