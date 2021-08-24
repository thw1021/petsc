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
#include <unordered_set>
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

struct ExecutionContext;
struct Edge;
class  Operator;
class  CallNode;
class  CallGraph;

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

struct Edge
{
  CallNode *const _begin;
  CallNode *const _end;
  PetscInt        _stream;


  constexpr Edge(CallNode *begin, CallNode *end, PetscInt stream = 0) noexcept
    : _begin(begin), _end(end), _stream(stream) { }
};

enum class OperatorKind : int {
  EMPTY,
  CALLABLE,
  COMPOSITION,
  CONDITIONAL
};

inline std::ostream& operator<<(std::ostream &strm, OperatorKind kind)
{
  switch (kind) {
  case OperatorKind::EMPTY:       return strm<<"OPERATOR_KIND_EMPTY";
  case OperatorKind::CALLABLE:    return strm<<"OPERATOR_KIND_CALLABLE";
  case OperatorKind::COMPOSITION: return strm<<"OPERATOR_KIND_COMPOSITION";
  case OperatorKind::CONDITIONAL: return strm<<"OPERATOR_KIND_CONDITIONAL";
  default:                        return strm;
  }
}

enum class NodeState : int {
  DISABLED,
  ENABLED,
  PLACEHOLDER
};

inline std::ostream& operator<<(std::ostream &strm, NodeState state)
{
  switch (state) {
  case NodeState::ENABLED:     return strm<<"NODE_STATE_ENABLED";
  case NodeState::DISABLED:    return strm<<"NODE_STATE_DISABLED";
  case NodeState::PLACEHOLDER: return strm<<"NODE_STATE_PLACEHOLDER";
  default:                     return strm;
  }
}

class Operator
{
public:
  using ptr_t = std::unique_ptr<const Operator>;

  virtual ~Operator() = default;

  virtual PetscErrorCode operator()(ExecutionContext*) const = 0;

  virtual OperatorKind kind() const = 0;

  ptr_t clone() const { return ptr_t(__cloneDerived());}

  virtual PetscErrorCode orderCallBack(const Edge*) const { return 0; }

  virtual NodeState defaultNodeState() const { return NodeState::DISABLED; }

protected:
  virtual Operator* __cloneDerived() const = 0;
};

class BranchOperator final : public Operator
{
public:
  using signature_t = PetscErrorCode(ExecutionContext*,const std::vector<CallNode*>&,PetscInt&);
  using function_t  = std::function<signature_t>;
  using self_t      = BranchOperator;

  template <typename T>
  BranchOperator(T &&fn) : _functor(std::forward<T>(fn)) { }

  PetscErrorCode operator()(ExecutionContext*) const override final;

  OperatorKind kind() const override final { return OperatorKind::CONDITIONAL; }

  PetscErrorCode orderCallBack(const Edge *edge) const override final
  {
    PetscFunctionBegin;
    _branches.insert(edge->_end);
    PetscFunctionReturn(0);
  }

  NodeState defaultNodeState() const override final { return NodeState::ENABLED; }

private:
  using set_t = std::unordered_set<CallNode*>;

  self_t* __cloneDerived() const override final { return new self_t(*this); }

  const function_t _functor;
  mutable set_t    _branches;
};

template <typename T, typename... Args>
class FunctionOperator final : public Operator
{
  template <bool val = true> struct is_wrapped_tag { static constexpr bool value = val; };
  template <> struct is_wrapped_tag<false> { static constexpr bool value = false; };

  using wrapped     = is_wrapped_tag<true>;
  using not_wrapped = is_wrapped_tag<false>;
  using is_wrapped  = is_wrapped_tag<is_invocable<T,ExecutionContext*,Args...>::value>;
  static constexpr bool is_wrapped_v = is_wrapped::value;

public:
  using signature_t = conditional_t<is_wrapped_v,
                                    PetscErrorCode(ExecutionContext*,Args...),
                                    PetscErrorCode(Args...)
                                    >;
  using function_t  = std::function<signature_t>;
  using self_t      = FunctionOperator<T,Args...>;

  FunctionOperator(T &&fn, std::tuple<Args...> &&args)
    : _functor(std::forward<T>(fn)), _params(std::move(args))
  { std::cout<<(is_wrapped_v ? "wrapped ctor\n" : "native ctor\n"); }

  PetscErrorCode operator()(ExecutionContext *ctx) const override final
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __callFunc(ctx,make_index_sequence<sizeof...(Args)>(),is_wrapped());CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  OperatorKind kind() const override final { return OperatorKind::CALLABLE; }

  NodeState defaultNodeState() const override final { return NodeState::ENABLED; }

private:
  const function_t          _functor;
  const std::tuple<Args...> _params;

  self_t* __cloneDerived() const override final { return new self_t(*this); }

  // wrapped call, where we need to pass the execution context through
  template <std::size_t... S>
  PetscErrorCode __callFunc(ExecutionContext *ctx, index_sequence<S...>, wrapped) const
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = _functor(ctx,std::get<S>(_params)...);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  template <std::size_t... S>
  PetscErrorCode __callFunc(PETSC_UNUSED ExecutionContext *ctx, index_sequence<S...>, not_wrapped) const
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = _functor(std::get<S>(_params)...);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
};

class CallNode
{
public:
  // Default constructor
  CallNode() = default;

  // Named constructor
  CallNode(std::string &&name) : _name(std::move(name)) { }
  explicit CallNode(const char name[]) : CallNode(std::string(name)) { }

  // Copy constructor
  CallNode(const CallNode &other)
    :  _inedges(other._inedges), _outedges(other._outedges),
       _operator(other._operator ? other._operator->clone() : nullptr), _name(other._name)
  { }

  // Move constructor
  CallNode(CallNode &&other) noexcept = default;

  // Composed constructor defined out of line as it needs CallGraph definition
  CallNode(CallGraph &graph);

  // The final constructor when constructing a fully built node
  CallNode(Operator *op) : _operator(op), _state(NodeState::ENABLED) { }

  // Constructs a function node.
  template <typename T, typename ...Args>
  CallNode(T &&f, std::tuple<Args...> &&args)
    : CallNode(new FunctionOperator<T,Args...>(std::forward<T>(f),std::forward<decltype(args)>(args)))
  { }

  // Templated constructor with bare arguments, mostly used to forward the arguments as a
  // tuple to the function operator constructor
  template <typename T, typename ...Args, enable_if_t<!std::is_convertible<T,Operator*>::value>* = nullptr>
  CallNode(T &&f, Args&&... args) : CallNode(std::forward<T>(f),std::forward_as_tuple(args...)) { }

  // Destructor
  ~CallNode() { std::cout<<"node "<<_id<<" dtor ("<<_name<<")\n"; }

  // Copy assignment operator
  CallNode& operator=(const CallNode&);

  // Private member getters
  PetscInt id() const { return _id; }
  const std::string& name() const { return _name; }
  NodeState state() const { return _state; }

  // Private member setters
  PetscErrorCode setName(const std::string &name)
  {
    PetscFunctionBegin;
    _name = name;
    PetscFunctionReturn(0);
  }

  PetscErrorCode setState(NodeState state)
  {
    PetscFunctionBegin;
    _state = state;
    CHKERRCXX(std::cout<<"node "<<_id<<" ("<<_name<<") state set: "<<state<<'\n');
    PetscFunctionReturn(0);
  }

  // Order enforcement
  PetscErrorCode before(CallNode*);
  template <std::size_t N>
  PetscErrorCode before(std::array<CallNode*,N>&);
  PetscErrorCode after(CallNode*);
  template <std::size_t N>
  PetscErrorCode after(std::array<CallNode*,N>&);

  PetscErrorCode view() const;

  // Execution
  PetscErrorCode run(ExecutionContext*) const;

private:
  friend class CallGraph;

  using edge_t = std::shared_ptr<Edge>;
  std::vector<edge_t> _inedges;
  std::vector<edge_t> _outedges;

  Operator::ptr_t _operator = nullptr;
  std::string     _name     = "anonymous node";
  NodeState       _state    = NodeState::DISABLED;
  const PetscInt  _id       = __counter(); // = counter++;

  // if you can figure out how to not get linker errors when using a static data member in
  // the ctors then go ahead and nix this useless function
  static PetscInt __counter()
  {
    static PetscInt counter = 0;
    return counter++;
  }
};

class CallGraph
{
public:
  // Default constructor
  CallGraph(std::string &&name = "anonymous graph")
    : _name(name),_begin(_name+" closure begin"),_end(_name+" closure end")
  {
    _begin._state = _end._state = NodeState::PLACEHOLDER;
  }

  explicit CallGraph(const char name[]) : CallGraph(std::string(name)) { }

  // Destructor
  ~CallGraph()
  {
    for (auto cnode : _graph) delete cnode;
    std::cout<<_name<<" dtor\n";
  }

  // Private member getters
  const std::string& name() const { return _name;}

  // Private member setters
  PetscErrorCode setName(const std::string &name)
  {
    PetscFunctionBegin;
    _name = name;
    _begin._name = _name+" closure begin";
    _end._name   = _name+" closure end";
    PetscFunctionReturn(0);
  }

  PetscErrorCode setUserContext(void *ctx)
  {
    PetscFunctionBegin;
    _userCtx = ctx;
    PetscFunctionReturn(0);
  }

  // Node emplacement
  template <typename T>
  CallNode* emplaceFunctionOperator(T&&);

  template <typename... Args, enable_if_t<(sizeof...(Args)>1)>* = nullptr>
  std::array<CallNode*,sizeof...(Args)> emplaceFunctionOperator(Args&&...);

  template <typename T, typename... Argr>
  CallNode* emplaceDirectFunctionOperator(T&&,Argr&&...);

  template <typename T, typename... Args>
  CallNode* emplaceBranchOperator(T&&,Args&&...);

  PetscErrorCode run(PetscInt = -1);

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

  template <typename... T>
  PetscErrorCode __emplaceOperatorCommon(T&&... args)
  {
    PetscFunctionBegin;
    _finalized = PETSC_FALSE;
    CHKERRCXX(_graph.emplace_back(new CallNode(std::forward<T>(args)...)));
    CHKERRCXX(_idMap[_graph.back()->_id] = _graph.size()-1);
    PetscFunctionReturn(0);
  }

  PetscErrorCode __topologicalSort(PetscInt v, std::vector<PetscBool> &visited)
  {
    PetscFunctionBegin;
    visited[v] = PETSC_TRUE;
    for (const auto &edge : _graph[v]->_inedges) {
      if (!visited[_idMap[edge->_begin->_id]]) {
        PetscErrorCode ierr;

        ierr = __topologicalSort(_idMap[edge->_begin->_id],visited);CHKERRQ(ierr);
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
      auto &vec = _graph[_idMap[edge->_end->_id]]->_inedges;

      CHKERRCXX(vec.erase(std::remove_if(vec.begin(),vec.end(),
      [&](const CallNode::edge_t &found)
      {
        return edge == found;
      })));
    }
    CHKERRCXX(_begin._outedges.clear());
    for (const auto &edge : _end._inedges) {
      auto &vec = _graph[_idMap[edge->_begin->_id]]->_outedges;

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

      ierr = __resetClosure();CHKERRQ(ierr);
      CHKERRCXX(_exec.clear());
      CHKERRCXX(_exec.reserve(graphSize+2));
      // install the start of the closure
      CHKERRCXX(_exec.push_back(&_begin));
      for (PetscInt i = 0; i < graphSize; ++i) {
        if (!visited[i]) {ierr = __topologicalSort(i,visited);CHKERRQ(ierr);}
        // if the node has no inedges it is a "start" node
        if (_graph[i]->_inedges.empty()) {ierr = _graph[i]->after(&_begin);CHKERRQ(ierr);}
        // if the node has no outedges it is an "end" node
        if (_graph[i]->_outedges.empty()) {ierr = _graph[i]->before(&_end);CHKERRQ(ierr);}
      }
      ierr = _begin.view();CHKERRQ(ierr);
      ierr = _end.view();CHKERRQ(ierr);
      // finish the closure
      CHKERRCXX(_exec.push_back(&_end));
      _finalized = PETSC_TRUE;
    }
    PetscFunctionReturn(0);
  }

  static PetscErrorCode __joinAncestors(CallNode &node, ExecutionContext &ctx)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = node.view();CHKERRQ(ierr);
    if (node._inedges.empty()) {
      // this node has no ancestors and is therefore *a* starting node of the graph
      std::cout<<"no ancestors\n";
      // however if the ctx is enclosed then it should not create a new stream, as the
      // enclosing scope has already set it
      if (!ctx.enclosed) ctx.stream = ctx.newStream();
    } else {
      PetscInt numCancelled = node._inedges[0]->_begin->_state == NodeState::DISABLED ? 1 : 0;
      // we have ancestors, meaning we must pick one of their streams and join all others
      // on that stream. we arbitrarily choose the first ancestor as the stream donor.
      ctx.stream = node._inedges[0]->_stream;
      // destroy all other streams except for the one we want to use. In the case we have
      // only 1 ancestor, this loop never fires
      CHKERRCXX(std::for_each(std::next(node._inedges.cbegin()),node._inedges.cend(),
      [&](const CallNode::edge_t &edge)
      {
        PetscFunctionBegin;
        if (edge->_begin->_state == NodeState::DISABLED) ++numCancelled;
        CHKERRCXX(std::cout<<"-- joining ancestor stream "<<edge->_stream<<'\n');
        PetscFunctionReturn(0);
      }));
      // only if all ancestors of a node are cancelled do we truly want to cancel it
      std::cout<<numCancelled<<'\n';
      if (numCancelled == node._inedges.size()) {
        CHKERRCXX(std::cout<<"All ancestors were cancelled, disabling node!\n");
        ierr = node.setState(NodeState::DISABLED);CHKERRQ(ierr);
      }
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
};

inline PetscErrorCode BranchOperator::operator()(ExecutionContext *exec) const
{
  const std::vector<CallNode*> branches(_branches.cbegin(),_branches.cend());
  PetscInt                     idx = -1;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = _functor(exec,branches,idx);CHKERRQ(ierr);
  if (idx >= 0) {
    for (PetscInt i = 0; i < idx; ++i) {
      CHKERRCXX(std::cout << "cancelling node " << branches[i]->id() << '\n');
      ierr = branches[i]->setState(NodeState::DISABLED);CHKERRQ(ierr);
    }
    for (PetscInt i = idx+1; i < branches.size(); ++i) {
      CHKERRCXX(std::cout<<"cancelling node "<<branches[i]->id()<<'\n');
      ierr = branches[i]->setState(NodeState::DISABLED);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

inline CallNode::CallNode(CallGraph &graph)
  : CallNode([&](ExecutionContext *ctx) { return graph.run(ctx->stream); })
{
  std::cout<<"COMPOSITION CTOR\n";
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = other->after(this);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::after(CallNode *other)
{
  PetscFunctionBegin;
  CHKERRCXX(std::cout<<_id<<" ("<<_name<<") after "<<other->_id<<" ("<<other->_name<<")\n");
  CHKERRCXX(_inedges.push_back(std::make_shared<Edge>(other,this)));
  CHKERRCXX(other->_outedges.push_back(_inedges.back()));
  if (other->_operator) {
    PetscErrorCode ierr;
    ierr = other->_operator->orderCallBack(other->_outedges.back().get());CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::view() const
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    CHKERRCXX(std::cout<<"node "<<_id<<"\nname: \""<<_name<<"\"\nstate: "<<_state<<"\ndeps: [");
    if (_inedges.empty()) CHKERRCXX(std::cout<<"none");
    else {
      for (const auto &in : _inedges) {
        CHKERRCXX(std::cout<<in->_begin->_id<<(in != _inedges.back() ? ", " : ""));
      }
    }
    CHKERRCXX(std::cout<<"] ----> ["<<_id<<" (this)] ----> [");
    if (_outedges.empty()) CHKERRCXX(std::cout<<"none]\n");
    else {
      for (const auto &out : _outedges) {
        CHKERRCXX(std::cout<<out->_end->_id<<(out != _outedges.back() ? ", " : "]\n"));
      }
    }
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::run(ExecutionContext *ctx) const
{
  PetscFunctionBegin;
  CHKERRCXX(std::cout<<_state<<'\n');
  if (_state == NodeState::ENABLED) {
    PetscErrorCode ierr;

    CHKERRCXX(std::cout<<"- running on stream "<<ctx->stream<<":\n");
    ierr = (*_operator)(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

inline CallNode& CallNode::operator=(const CallNode &other)
{
  PetscFunctionBegin;
  if (this != &other) {
    // this is to safeguard against:
    // 1. you create a node normally in a graph, it has prerequisites, dependencies and
    //    everything is A-ok.
    // 2. you create a completely separate node, possibly in a completely separate graph,
    //    which also has some dependencies.
    // 3. you now copy assign one to the other, and try to run one of the graphs. The
    //    graph tries to linearize the DAG but now the node you copy assigned to has
    //    dependencies in a completely separate graph! What do?
    if (PetscUnlikelyDebug(_operator && (_operator->kind() != OperatorKind::EMPTY))) SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot copy assign to a node which already has a valid OperatorKind");
    if (other._operator) _operator = other._operator->clone();
    _inedges  = other._inedges;
    _outedges = other._outedges;
    _name     = other._name;
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline CallNode* CallGraph::emplaceFunctionOperator(T &&fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // we allow CallGraph through this check as it uses CallGraph::run() as the canonical
  // operator(). We could instead explicitly overload this routine for CallGraph&, but
  // that is a lot code bloat for just one line
  static_assert(std::is_same<decay_t<T>,CallGraph>::value || is_callable<T>::value,"");
  ierr = __emplaceOperatorCommon(std::forward<T>(fn));CHKERRABORT(PETSC_COMM_SELF,ierr);
  PetscFunctionReturn(_graph.back());
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 1)>*>
inline std::array<CallNode*,sizeof...(Args)> CallGraph::emplaceFunctionOperator(Args&&... rest)
{
  _graph.reserve(_graph.size()+sizeof...(Args));
  return {emplaceFunctionOperator(std::forward<Args>(rest))...};
}

template <typename T, typename... Argr>
inline CallNode* CallGraph::emplaceDirectFunctionOperator(T &&f, Argr&&... args)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  static_assert((sizeof...(Argr) > 0) && is_callable<T>::value,"");
  ierr = __emplaceOperatorCommon(std::forward<T>(f),std::forward<Argr>(args)...);CHKERRABORT(PETSC_COMM_SELF,ierr);
  PetscFunctionReturn(_graph.back());
}

template <typename T, typename... Args>
inline CallNode* CallGraph::emplaceBranchOperator(T&& f, Args&&... nodes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  static_assert(sizeof...(Args) == 0 || all_same<CallNode*,Args...>::value,"");
  ierr = __emplaceOperatorCommon(new BranchOperator(std::forward<T>(f),std::forward<Args>(nodes)...));CHKERRABORT(PETSC_COMM_SELF,ierr);
  PetscFunctionReturn(_graph.back());
}

inline PetscErrorCode CallGraph::run(PetscInt stream)
{
  ExecutionContext ctx(_userCtx,stream);
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = __finalize();CHKERRQ(ierr);
  CHKERRCXX(std::cout<<"----- "<<_name<<" begin run\n");
  for (PetscInt i = 0; i < _exec.size(); ++i) {
    ierr = __joinAncestors(*_exec[i],ctx);CHKERRQ(ierr);
    ierr = _exec[i]->run(&ctx);CHKERRQ(ierr);
    ierr = __forkDecendants(*_exec[i],ctx);CHKERRQ(ierr);
  }
  // re-enable any nodes that were disabled as part of the run
  for (auto &node : _graph) {
    ierr = node->setState(node->_operator->defaultNodeState());CHKERRQ(ierr);
  }
  CHKERRCXX(std::cout<<"----- "<<_name<<" end run\n");
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCGRAPH_H */
