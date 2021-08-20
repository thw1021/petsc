#include "petscgraph.hpp"

namespace Petsc {

PetscInt ExecutionContext::pool = 0;

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

PetscErrorCode CallNode::after(CallNode &other)
{
  PetscFunctionBegin;
  std::cout<<_id<<" after "<<other._id<<std::endl;
  CHKERRCXX(_inedges.push_back(std::make_shared<Edge>(other._id,_id)));
  CHKERRCXX(other._outedges.push_back(_inedges.back()));
  PetscFunctionReturn(0);
}

PetscErrorCode CallNode::run(ExecutionContext *ctx) const
{
  PetscFunctionBegin;
  if (_functor) {
    PetscErrorCode ierr;

    std::cout<<"- running on stream "<<ctx->stream<<":\n";
    ierr = (*_functor)(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
  ierr = __finalize();CHKERRQ(ierr);
  std::cout<<"----- "<<_name<<" begin run\n";
  for (const auto &node : _exec) {
    ierr = node->__printDeps();CHKERRQ(ierr);
    ierr = __joinAncestors(*node,ctx);CHKERRQ(ierr);
    ierr = node->run(&ctx);CHKERRQ(ierr);
    ierr = __forkDecendants(*node,ctx);CHKERRQ(ierr);
  }
  std::cout<<"----- "<<_name<<" end run\n";
  PetscFunctionReturn(0);
}

}

PetscErrorCode PetscCallGraphCreate(MPI_Comm comm, PetscCallGraph *graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
