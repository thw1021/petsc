#ifndef PETSCGRAPH_H
#define PETSCGRAPH_H

#include <petscsys.h>
#include <petscgraphtypes.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/traithelpers.hpp>

PETSC_EXTERN PetscErrorCode PetscCallNodeCreate(MPI_Comm,PetscCallNode*);
PETSC_EXTERN PetscErrorCode PetscCallNodeDestroy(PetscCallNode*);
PETSC_EXTERN PetscErrorCode PetscCallNodeView(PetscCallNode,PetscViewer);

PETSC_EXTERN PetscErrorCode PetscCallGraphCreate(MPI_Comm,PetscCallGraph*);
PETSC_EXTERN PetscErrorCode PetscCallGraphDestroy(PetscCallGraph*);
PETSC_EXTERN PetscErrorCode PetscCallGraphView(PetscCallGraph,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscCallGraphAddNode(PetscCallGraph,PetscCallNode);

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

#if 1
#  define CXXPRINT(message) CHKERRCXX(std::cout<<message)
#else
#  define CXXPRINT(message)
#endif

PETSC_CXX_ENUM_WRAPPER_DECLARE(OperatorKind,int,EMPTY,CALLABLE,COMPOSITION,CONDITIONAL);

PETSC_CXX_ENUM_WRAPPER_DECLARE(NodeState,int,DISABLED,ENABLED,PLACEHOLDER);

struct ExecutionContext;
struct Edge;
class  Operator;
class  CallNode;
class  CallGraph;

struct ExecutionContext
{
  void *const        _userCtx;
  PetscDeviceContext _dctx;
  const PetscBool    _enclosed;

  constexpr ExecutionContext(void *ctx, PetscDeviceContext strm, PetscBool enclosed) noexcept
    : _userCtx(ctx), _dctx(strm), _enclosed(enclosed)
  { }

  PETSC_NODISCARD PetscErrorCode getUserContext(void *ctx) const
  {
    PetscFunctionBegin;
    PetscValidPointer(ctx,1);
    *static_cast<void**>(ctx) = _userCtx;
    PetscFunctionReturn(0);
  }
};

struct Edge
{
  CallNode *const    _begin;
  CallNode *const    _end;
  PetscDeviceContext _dctx;

  constexpr Edge(CallNode *begin, CallNode *end, PetscDeviceContext ctx = nullptr) noexcept
    : _begin(begin), _end(end), _dctx(ctx)
  { }
};

class Operator
{
public:
  using pointer_type = std::unique_ptr<const Operator>;

  virtual ~Operator() = default;

  PETSC_NODISCARD virtual PetscErrorCode operator()(ExecutionContext*) const = 0;

  PETSC_NODISCARD virtual OperatorKind kind() const = 0;

  // virtual-final to prevent derived classes overriding or hiding this function. as an
  // added benefit since this class doesn't inherit from anyone else, this function doesn't get
  // put in the vtable, and can be inlined directly
  PETSC_NODISCARD virtual pointer_type clone() const final { return pointer_type(__cloneDerived()); }

  PETSC_NODISCARD virtual PetscErrorCode orderCallBack(const std::shared_ptr<const Edge>&) const
  { return 0; }

  PETSC_NODISCARD virtual NodeState defaultNodeState() const { return NodeState::DISABLED; }

protected:
  PETSC_NODISCARD virtual Operator* __cloneDerived() const = 0;
};

class BranchOperator final : public Operator
{
public:
  using signature_type = PetscErrorCode(ExecutionContext*,const std::vector<CallNode*>&,PetscInt&);
  using function_type  = std::function<signature_type>;
  using container_type = std::unordered_set<CallNode*>;
  using self_type      = BranchOperator;

  template <typename T>
  constexpr BranchOperator(T &&fn) : _functor(std::forward<T>(fn)) { }

  PETSC_NODISCARD PetscErrorCode operator()(ExecutionContext*) const override final;

  PETSC_NODISCARD OperatorKind kind() const override final { return OperatorKind::CONDITIONAL; }

  PETSC_NODISCARD PetscErrorCode orderCallBack(const std::shared_ptr<const Edge> &edge) const override final
  {
    PetscFunctionBegin;
    CHKERRCXX(_branches.insert(edge->_end));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD NodeState defaultNodeState() const override final { return NodeState::ENABLED; }

private:
  const function_type    _functor;
  mutable container_type _branches;

  PETSC_NODISCARD self_type* __cloneDerived() const override final { return new self_type(*this); }
};

template <typename T, typename... Args>
class FunctionOperator final : public Operator
{
  template <bool val = true> struct is_wrapped_tag { static constexpr bool value = val; };
  template <> struct is_wrapped_tag<false> { static constexpr bool value = false; };

  using is_wrapped  = is_wrapped_tag<is_invocable<T,ExecutionContext*,Args...>::value>;
  using wrapped     = is_wrapped_tag<true>;
  using not_wrapped = is_wrapped_tag<false>;

public:
  using signature_type = conditional_t<is_wrapped::value,
                                       PetscErrorCode(ExecutionContext*,Args...),
                                       PetscErrorCode(Args...)
                                       >;
  using function_type  = std::function<signature_type>;
  using container_type = std::tuple<Args...>;
  using self_type      = FunctionOperator<T,Args...>;

  constexpr FunctionOperator(T &&fn, std::tuple<Args...> &&args)
    : _functor(std::forward<T>(fn)), _params(std::move(args))
  {
    // so that the static_assert error message is intelligible
    using functionSignatureMatchesArgumentList = conditional_t<is_wrapped::value,is_invocable<T,ExecutionContext*,Args...>,is_invocable<T,Args...>>;

    static_assert(functionSignatureMatchesArgumentList::value,"Function does not appear to be invocable with the given argument list");
  }

  PETSC_NODISCARD PetscErrorCode operator()(ExecutionContext *ctx) const override final
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __callFunc(ctx,make_index_sequence<sizeof...(Args)>(),is_wrapped());CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD OperatorKind kind() const override final { return OperatorKind::CALLABLE; }

  PETSC_NODISCARD NodeState defaultNodeState() const override final { return NodeState::ENABLED; }

private:
  const function_type  _functor;
  const container_type _params;

  PETSC_NODISCARD self_type* __cloneDerived() const override final { return new self_type(*this); }

  // wrapped call, where we need to pass the execution context through
  template <std::size_t... S>
  PETSC_NODISCARD PetscErrorCode __callFunc(ExecutionContext *ctx, index_sequence<S...>, wrapped) const
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = _functor(ctx,std::get<S>(_params)...);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  template <std::size_t... S>
  PETSC_NODISCARD PetscErrorCode __callFunc(ExecutionContext*, index_sequence<S...>, not_wrapped) const
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
  CallNode() noexcept = default;

  // Named constructor
  template <typename T,
            enable_if_t<std::is_convertible<remove_cvref_t<T>,std::string>::value>* = nullptr>
  CallNode(T&& name) : _name(std::forward<T>(name)) { }

  // Copy constructor
  CallNode(const CallNode &other)
    :  _inedges(other._inedges), _outedges(other._outedges),
       _operator(other._operator ? other._operator->clone() : nullptr), _name(other._name),
       _state(other._state)
  { }

  // Move constructor
  CallNode(CallNode &&other) noexcept = default;

  // Composed constructor defined out of line as it needs CallGraph definition
  CallNode(CallGraph &graph);

  // The final constructor when constructing a fully built node
  CallNode(const Operator *op) : _operator(op), _state(op->defaultNodeState()) { }

  // Constructs a function node.
  template <typename T, typename ...Args>
  constexpr CallNode(T &&f, std::tuple<Args...> &&args)
    : CallNode(new FunctionOperator<T,Args...>(std::forward<T>(f),std::forward<decltype(args)>(args)))
  { }

  // Templated constructor with bare arguments, mostly used to forward the arguments as a
  // tuple to the function operator constructor
  template <typename T, typename ...Args,
            enable_if_t<!std::is_convertible<T,Operator*>::value>* = nullptr>
  CallNode(T &&f, Args&&... args) : CallNode(std::forward<T>(f),std::forward_as_tuple(args...)) { }

  // Destructor
  ~CallNode() { std::cout<<"node "<<_id<<" dtor ("<<_name<<std::endl; }

  // Private member getters
  PETSC_NODISCARD constexpr PetscInt id() const { return _id; }
  PETSC_NODISCARD constexpr const std::string& name() const { return _name; }
  PETSC_NODISCARD constexpr NodeState state() const { return _state; }

  // Private member setters
  PETSC_NODISCARD PetscErrorCode setName(const std::string &name)
  {
    PetscFunctionBegin;
    CXXPRINT("node "<<_id<<" ("<<_name<<") new name set: "<<name<<'\n');
    _name = name;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode setState(NodeState state)
  {
    PetscFunctionBegin;
    _state = state;
    CXXPRINT("node "<<_id<<" ("<<_name<<") new state set: "<<state<<'\n');
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode resetState()
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = this->setState(_operator ? _operator->defaultNodeState() : NodeState::DISABLED);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  // Order enforcement
  PETSC_NODISCARD PetscErrorCode before(CallNode*);
  template <std::size_t N>
  PETSC_NODISCARD PetscErrorCode before(const std::array<CallNode*,N>&);
  PETSC_NODISCARD PetscErrorCode after(CallNode*);
  template <std::size_t N>
  PETSC_NODISCARD PetscErrorCode after(const std::array<CallNode*,N>&);

  PETSC_NODISCARD PetscErrorCode view() const;

  // Execution
  PETSC_NODISCARD PetscErrorCode run(ExecutionContext*) const;

  // Copy assignment operator
  CallNode& operator=(const CallNode&);

private:
  friend class CallGraph;

  using edge_type = std::shared_ptr<Edge>;
  std::vector<edge_type> _inedges;
  std::vector<edge_type> _outedges;

  using operator_type = Operator::pointer_type;
  operator_type  _operator = nullptr;
  std::string    _name     = "anonymous node";
  NodeState      _state    = NodeState::DISABLED;
  const PetscInt _id       = __counter(); // = counter++;

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
  CallGraph() : CallGraph(std::string()) { }

  // Named constructor
  template <typename T,
            enable_if_t<std::is_convertible<remove_cvref_t<T>,std::string>::value>* = nullptr>
  CallGraph(T &&name)
    : _name(std::forward<T>(name)),_begin(_name+" closure begin"), _end(_name+" closure end")
  {
    _begin._state = _end._state = NodeState::PLACEHOLDER;
  }

  // Destructor
  ~CallGraph()
  {
    for (auto cnode : _graph) delete cnode;
    std::cout<<_name<<" dtor\n";
  }

  // Private member getters
  PETSC_NODISCARD const std::string& name() const { return _name;}

  // Private member setters
  PETSC_NODISCARD PetscErrorCode setName(const std::string &name)
  {
    PetscFunctionBegin;
    _name        = name;
    _begin._name = name+" closure begin";
    _end._name   = name+" closure end";
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode setUserContext(void *ctx)
  {
    PetscFunctionBegin;
    PetscValidPointer(ctx,1);
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

  template <typename T>
  CallNode* emplaceBranchOperator(T&&);

  PETSC_NODISCARD PetscErrorCode run(PetscDeviceContext cxt = nullptr);

private:
  using container_type = std::vector<CallNode*>;
  using map_type       = std::unordered_map<PetscInt,PetscInt>;

  std::string     _name;
  map_type        _idMap;
  CallNode        _begin;
  CallNode        _end;
  container_type  _graph;
  container_type  _exec      = {&_begin};
  PetscBool       _finalized = PETSC_FALSE;
  void           *_userCtx   = nullptr;

  template <typename... T>
  PETSC_NODISCARD PetscErrorCode __emplaceOperatorCommon(T&&... args)
  {
    bool success;

    PetscFunctionBegin;
    _finalized = PETSC_FALSE;
    CHKERRCXX(_graph.emplace_back(new CallNode(std::forward<T>(args)...)));
    // success = false if the key already existed
    CHKERRCXX(std::tie(std::ignore,success) = _idMap.insert({_graph.back()->_id,_graph.size()-1}));
    if (PetscUnlikelyDebug(!success)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Attempted to insert duplicate node into graph");
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode __topologicalSort(PetscInt v, std::vector<PetscBool> &visited)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    visited[v] = PETSC_TRUE;
    for (const auto &edge : _graph[v]->_inedges) {
      if (!visited[_idMap[edge->_begin->_id]]) {
        ierr = __topologicalSort(_idMap[edge->_begin->_id],visited);CHKERRQ(ierr);
      }
    }
    CHKERRCXX(_exec.push_back(_graph[v]));
    PetscFunctionReturn(0);
  }

  template <typename T, typename T2>
  PETSC_NODISCARD static PetscErrorCode __removeItemFrom(const T2 &item, T &container)
  {
    using std::begin;
    using std::end;  // enable ADL
    const auto found = [&](const T2 &it) { return it == item; };

    PetscFunctionBegin;
    CHKERRCXX(container.erase(std::remove_if(begin(container),end(container),found)));
    PetscFunctionReturn(0);
  }

  // this routine finds and removes the fake dependencies on the _begin and _end
  // nodes. Since these are shared_ptr's we need to delete them in both _begin and _end
  // as well as the nodes they are attached to in _graph.
  PETSC_NODISCARD PetscErrorCode __resetClosure()
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    for (const auto &edge : _begin._outedges) {
      ierr = __removeItemFrom(edge,_graph[_idMap[edge->_end->_id]]->_inedges);CHKERRQ(ierr);
    }
    CHKERRCXX(_begin._outedges.clear());
    for (const auto &edge : _end._inedges) {
      ierr = __removeItemFrom(edge,_graph[_idMap[edge->_begin->_id]]->_outedges);CHKERRQ(ierr);
    }
    CHKERRCXX(_end._inedges.clear());
    CHKERRCXX(_exec.resize(1)); // the only callnode left should be _begin
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode __finalize()
  {
    PetscFunctionBegin;
    if (!_finalized) {
      std::vector<PetscBool> visited(_graph.size(),PETSC_FALSE);
      PetscErrorCode         ierr;

      ierr = __resetClosure();CHKERRQ(ierr);
      for (PetscInt i = 0; i < _graph.size(); ++i) {
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

  PETSC_NODISCARD static PetscErrorCode __joinAncestors(CallNode *node, ExecutionContext *exec)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = node->view();CHKERRQ(ierr);
    if (node->_inedges.size()) {
      PetscInt numCancelled = 0;

      for (PetscInt i = 0; i < node->_inedges.size(); ++i) {
        const auto &edge = node->_inedges[i];
        // we have ancestors, meaning we must pick one of their streams and join all others
        // on that stream.
        if (i) {
          ierr = PetscDeviceContextWaitForContext(exec->_dctx,edge->_dctx);CHKERRQ(ierr);
          ierr = PetscDeviceContextDestroy(&edge->_dctx);CHKERRQ(ierr);
        } else {
          // we enforce strict primogeniture for the base stream inheritance, this is to ensure
          // that when the streams are joined the correct parent stream is always selected
          exec->_dctx = edge->_dctx;
        }
        numCancelled += (edge->_begin->_state == NodeState::DISABLED);
      }
      // only if all ancestors of a node are cancelled do we truly want to cancel it
      CXXPRINT(numCancelled<<'\n');
      if (numCancelled == node->_inedges.size()) {
        CXXPRINT("All ancestors were cancelled, disabling node!\n");
        ierr = node->setState(NodeState::DISABLED);CHKERRQ(ierr);
      }
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __forkDecendants(const CallNode *node, const ExecutionContext *exec)
  {
    PetscFunctionBegin;
    for (PetscInt i = 0; i < node->_outedges.size(); ++i) {
      const auto &edge = node->_outedges[i];
      // if this loop fires we have decendants, so we need to fork some streams off of the
      // current one, although we need only create n-1 streams since we can give ours away.
      if (i) {
        PetscErrorCode ierr;

        // everyone else gets duped and waits on donor stream
        ierr = PetscDeviceContextDuplicate(exec->_dctx,&edge->_dctx);CHKERRQ(ierr);
        ierr = PetscDeviceContextWaitForContext(edge->_dctx,exec->_dctx);CHKERRQ(ierr);
      } else {
        edge->_dctx = exec->_dctx;
      }
    }
    PetscFunctionReturn(0);
  }
};

inline PetscErrorCode BranchOperator::operator()(ExecutionContext *exec) const
{
  PetscInt       id = -1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = _functor(exec,{_branches.cbegin(),_branches.cend()},id);CHKERRQ(ierr);
  for (const auto &branch : _branches) {
    if (id != branch->id()) {
      CXXPRINT("cancelling node "<<branch->id()<<'\n');
      ierr = branch->setState(NodeState::DISABLED);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

inline CallNode::CallNode(CallGraph &graph)
  : CallNode([&](ExecutionContext *ctx) { return graph.run(ctx->_dctx); })
{ }

template <std::size_t N>
inline PetscErrorCode CallNode::before(const std::array<CallNode*,N> &others)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (const auto &other : others) {ierr = this->before(other);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template <std::size_t N>
inline PetscErrorCode CallNode::after(const std::array<CallNode*,N> &others)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (const auto &other : others) {ierr = this->after(other);CHKERRQ(ierr);}
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
  CXXPRINT("node "<<_id<<" ("<<_name<<") after "<<other->_id<<" ("<<other->_name<<")\n");
  CHKERRCXX(_inedges.push_back(std::make_shared<Edge>(other,this)));
  CHKERRCXX(other->_outedges.push_back(_inedges.back()));
  if (other->_operator) {
    PetscErrorCode ierr;
    ierr = other->_operator->orderCallBack(_inedges.back());CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::view() const
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    CXXPRINT("node "<<_id<<"\nname: \""<<_name<<"\"\nstate: "<<_state<<"\ndeps: [");
    if (_inedges.empty()) CXXPRINT("none");
    else {
      for (const auto &in : _inedges) {
        CXXPRINT(in->_begin->_id<<(in != _inedges.back() ? ", " : ""));
      }
    }
    CXXPRINT("] ----> ["<<_id<<" (this)] ----> [");
    if (_outedges.empty()) CXXPRINT("none]\n");
    else {
      for (const auto &out : _outedges) {
        CXXPRINT(out->_end->_id<<(out != _outedges.back() ? ", " : "]\n"));
      }
    }
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode CallNode::run(ExecutionContext *ctx) const
{
  PetscFunctionBegin;
  CXXPRINT(_state<<'\n');
  if (_state == NodeState::ENABLED) {
    PetscErrorCode ierr;

    CXXPRINT("- running on stream "<<ctx->_dctx->id<<":\n");
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
    _operator = other._operator ? other._operator->clone() : nullptr;
    _inedges  = other._inedges;
    _outedges = other._outedges;
    _name     = other._name;
    _state    = other._state;
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

template <typename T>
inline CallNode* CallGraph::emplaceBranchOperator(T&& f)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = __emplaceOperatorCommon(new BranchOperator(std::forward<T>(f)));CHKERRABORT(PETSC_COMM_SELF,ierr);
  PetscFunctionReturn(_graph.back());
}

inline PetscErrorCode CallGraph::run(PetscDeviceContext ctx)
{
  const PetscBool enclosed = ctx ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (enclosed) {
    PetscValidDeviceContext(ctx,1);
  } else {
    ierr = PetscDeviceContextCreate(&ctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetStreamType(ctx,PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetUp(ctx);CHKERRQ(ierr);
  }
  ierr = __finalize();CHKERRQ(ierr);
  CXXPRINT("----- "<<_name<<" begin run\n");
  {
    ExecutionContext exec(_userCtx,ctx,enclosed);

    for (const auto &enode : _exec) {
      ierr = __joinAncestors(enode,&exec);CHKERRQ(ierr);
      ierr = enode->run(&exec);CHKERRQ(ierr);
      ierr = __forkDecendants(enode,&exec);CHKERRQ(ierr);
    }
  }
  // reset any nodes that were disabled as part of the run
  for (const auto &node : _graph) {ierr = node->resetState();CHKERRQ(ierr);}
  CXXPRINT("----- "<<_name<<" end run\n");
  if (!enclosed) {
    ierr = PetscDeviceContextSynchronize(ctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextDestroy(&ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCGRAPH_H */
