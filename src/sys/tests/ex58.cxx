#include <petsc/private/petscimpl.h>
#include <petscgraph.hpp>

using namespace Petsc;

struct UserCtx
{
  PetscInt value;

  UserCtx(PetscInt val) : value(val) { }
};

PetscErrorCode startFunc(ExecutionContext *exec)
{
  static PetscBool rep = PETSC_TRUE;
  UserCtx          *ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = exec->getUserContext((void**)&ctx);CHKERRQ(ierr);
  ierr = exec->repeat(rep);CHKERRQ(ierr);
  rep  = PETSC_FALSE;
  std::cout<<"start node, value = "<<ctx->value<<'\n';
  PetscFunctionReturn(0);
}

PetscErrorCode joinNodeFunction(ExecutionContext *ctx, PetscInt x)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  auto context = ctx->userCtx;
  (void)(context);
  std::cout<<"join node, value = "<<x<<'\n';
  PetscFunctionReturn(0);
}

PetscErrorCode testFuncOtherGraph(ExecutionContext *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  std::cout<<"other graph\n";
  PetscFunctionReturn(0);
}

PetscErrorCode nativeFunction(PetscInt *x)
{
  PetscFunctionBegin;
  std::cout<<"native function "<<*x<<'\n';
  *x = 5;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  PetscInt  x = 123456;
  UserCtx   ctx(3);
  CallGraph graph2("small graph");
  CallNode  node,node2 = CallNode();

  node2 = node;
  ierr = graph2.setUserContext(&ctx);CHKERRQ(ierr);
  auto smallGraphLeft    = graph2.emplaceFunctionOperator(testFuncOtherGraph);
  ierr = smallGraphLeft->setName("small graph left");CHKERRQ(ierr);
  auto smallGraphLeftDep = graph2.emplaceFunctionOperator([](ExecutionContext*)
  {
    CHKERRCXX(std::cout<<"running left dep\n");
    return 0;
  });
  ierr = smallGraphLeftDep->setName("small graph left dependency");CHKERRQ(ierr);
  ierr = smallGraphLeftDep->after(smallGraphLeft);CHKERRQ(ierr);
  auto smallGraphRight   = graph2.emplaceDirectFunctionOperator(nativeFunction,&x);
  ierr = smallGraphRight->setName("small graph right");CHKERRQ(ierr);

  auto branchNode = graph2.emplaceBranchOperator([](ExecutionContext *exec, const std::vector<CallNode*> &branches, PetscInt &idx)
  {
    PetscFunctionBegin;
    std::cout<<"choice of branches: ";
    for (const auto branch : branches) {
      std::cout<<branch->id()<<' ';
    }
    idx = 0;
    std::cout<<"\nchose branch #"<<idx<<" (id "<<branches[idx]->id()<<")\n";
    PetscFunctionReturn(0);
  });
  ierr = branchNode->setName("branch node");CHKERRQ(ierr);
  ierr = branchNode->before(smallGraphLeft);CHKERRQ(ierr);
  ierr = branchNode->before(smallGraphRight);CHKERRQ(ierr);

  std::cout<<"-------------------"<<std::endl;
  {
    PetscInt  n = 2;
    CallGraph graph;

    ierr = graph.setName("big graph");CHKERRQ(ierr);
    auto graphNode = graph.emplaceFunctionOperator(graph2);
    ierr = graphNode->setName("graph node");CHKERRQ(ierr);
    auto nodeStart = graph.emplaceFunctionOperator(startFunc);
    ierr = nodeStart->setName("node start");CHKERRQ(ierr);
    auto nodeJoin  = graph.emplaceDirectFunctionOperator(joinNodeFunction,n);
    ierr = nodeJoin->setName("join node");CHKERRQ(ierr);
    auto midNodes  = graph.emplaceFunctionOperator([](ExecutionContext *ctx)
    {
      std::cout<<"mid node left\n";
      return 0;
    }, [](ExecutionContext *ctx) {
      std::cout<<"mid node right\n";
      return 0;
    });
    ierr = midNodes[0]->setName("mid node left");CHKERRQ(ierr);
    ierr = midNodes[1]->setName("mid node right");CHKERRQ(ierr);
    auto forkNodes = graph.emplaceFunctionOperator([](ExecutionContext *ctx)
    {
      std::cout<<"fork node left\n";
      return 0;
    },[](ExecutionContext *ctx) {
      std::cout<<"fork node right\n";
      return 0;
    });
    ierr = forkNodes[0]->setName("fork node left");CHKERRQ(ierr);
    ierr = forkNodes[1]->setName("fork node right");CHKERRQ(ierr);
    ierr = nodeStart->before(std::get<0>(midNodes));CHKERRQ(ierr);
    ierr = nodeJoin->after(midNodes);CHKERRQ(ierr);
    ierr = nodeJoin->before(forkNodes);CHKERRQ(ierr);
    ierr = graphNode->after(forkNodes);CHKERRQ(ierr);

    ierr = graph.setUserContext(&ctx);CHKERRQ(ierr);
    ierr = graph.run();CHKERRQ(ierr);
  }
  ierr = graph2.run();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
