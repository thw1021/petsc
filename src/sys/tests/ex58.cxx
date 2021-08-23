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
  graph2.setUserContext(&ctx);
  graph2.emplace(testFuncOtherGraph);
  graph2.emplaceCall(nativeFunction,&x);

  BranchOperator branch([](ExecutionContext *exec, const std::vector<CallNode*> &branches, PetscInt &id)
  {
    PetscFunctionBegin;
    for (const auto branch : branches) {
      std::cout<<branch->id()<<'\n';
    }
    id = branches[0]->id();
    PetscFunctionReturn(0);
  },&node,&node2);

  std::cout<<"-------------------"<<std::endl;
  {
    PetscInt  n = 2;
    CallGraph graph("big graph");

    auto graphNode = graph.emplace(graph2);
    auto nodeStart = graph.emplace(startFunc);
    auto nodeJoin  = graph.emplaceCall(joinNodeFunction,n);
    auto midNodes  = graph.emplace([](ExecutionContext *ctx)
    {
      std::cout<<"mid node left\n";
      return 0;
    }, [](ExecutionContext *ctx) {
      std::cout<<"mid node right\n";
      return 0;
    });
    auto nodeFork = graph.emplace([](ExecutionContext *ctx)
    {
      std::cout<<"fork node left\n";
      return 0;
    },[](ExecutionContext *ctx) {
      std::cout<<"fork node right\n";
      return 0;
    });
    ierr = nodeStart->before(std::get<0>(midNodes));CHKERRQ(ierr);
    ierr = nodeJoin->after(midNodes);CHKERRQ(ierr);
    ierr = nodeJoin->before(nodeFork);CHKERRQ(ierr);
    ierr = graphNode->after(nodeFork);CHKERRQ(ierr);

    ierr = graph.setUserContext(&ctx);CHKERRQ(ierr);
    ierr = graph.run();CHKERRQ(ierr);
  }
  ierr = graph2.run();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
