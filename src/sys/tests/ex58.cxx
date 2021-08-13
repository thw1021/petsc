#include <petscgraph.hpp>

using namespace Petsc;

PetscInt CallNode::counter = 0;

struct UserCtx
{
  int value;

  UserCtx(int i) : value(i) { }
};

PetscErrorCode testFunc(ExecutionContext *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  auto context = ctx->userCtx;
  (void)(context);
  std::cout<<"start node\n";
  PetscFunctionReturn(0);
}

PetscErrorCode staticTestFunc(ExecutionContext *ctx, int x)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  auto context = ctx->userCtx;
  (void)(context);
  std::cout<<"join node\n";
  PetscFunctionReturn(0);
}

PetscErrorCode testFuncOtherGraph(ExecutionContext *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  std::cout<<"other graph"<<std::endl;
  PetscFunctionReturn(0);
}

PetscErrorCode nativeFunction(int x)
{
  PetscFunctionBegin;
  std::cout<<x<<std::endl;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  UserCtx   ctx(3);
  CallGraph graph2("small graph");
  CallNode  node,node2 = CallNode();

  node2 = node;
  graph2.setUserContext(&ctx);
  graph2.emplace(testFuncOtherGraph);
  graph2.emplaceCall(nativeFunction,2);
  std::cout<<"-------------------"<<std::endl;
  {
    CallGraph graph("big graph");
    CallNode  *graphNode;

    ierr = graph.compose(graph2,graphNode);CHKERRQ(ierr);
    auto nodeStart = graph.emplace(testFunc);
    auto nodeJoin  = graph.emplaceCall(staticTestFunc,2);
    auto midNodes  = graph.emplace([](ExecutionContext *ctx, int x = 2)
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
    ierr = nodeStart->before(*std::get<0>(midNodes));CHKERRQ(ierr);
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
