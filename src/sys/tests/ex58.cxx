#include <petscgraph.hpp>

using namespace Petsc;

PetscInt CallNode::counter = 0;

struct UserCtx
{
  int value;

  UserCtx(int i) : value(i) { }
};

PetscErrorCode testFunc(void *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  auto context = static_cast<UserCtx*>(ctx);
  PetscFunctionReturn(0);
}

PetscErrorCode staticTestFunc(void *ctx, int x)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx,1);
  auto context = static_cast<UserCtx*>(ctx);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  UserCtx ctx(3);
  CallGraph graph;
  CallNode  node,node2 = CallNode();

  node2 = node;
  std::cout<<"------"<<std::endl;
  auto nodeMiddle = graph.emplace(testFunc);
  auto nodeEnd    = graph.emplaceCall(staticTestFunc,2);
  auto otherNodes = graph.emplace([](void *ctx, int x = 2)
  {
    std::cout<<"lambda left branch"<<std::endl;
    return 0;
  }, [](void *ctx) {
    std::cout<<"lambda right branch"<<std::endl;
    return 0;
  });
  ierr = nodeMiddle->before(*std::get<1>(otherNodes));CHKERRQ(ierr);
  ierr = nodeEnd->after(otherNodes);CHKERRQ(ierr);

  ierr = graph.setUserContext(&ctx);CHKERRQ(ierr);
  ierr = graph.run();CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
