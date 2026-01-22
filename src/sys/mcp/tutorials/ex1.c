static char help[] = "Basic MCP Server Example.\n\n";

#include <petscmcp.h>

int main(int argc, char **argv)
{
  PetscMCP mcp;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscMCPCreate(PETSC_COMM_WORLD, &mcp));
  PetscCall(PetscMCPRun(mcp));
  PetscCall(PetscMCPDestroy(&mcp));

  PetscCall(PetscFinalize());
  return 0;
}
