#if !defined(PETSCMCP_H)
#define PETSCMCP_H

#include <petscsys.h>

/*S
   PetscMCP - Model Context Protocol (MCP) server context

   Level: developer

.seealso: `PetscMCPCreate()`, `PetscMCPRun()`, `PetscMCPDestroy()`
S*/
typedef struct _p_PetscMCP* PetscMCP;

/*J
   PetscMCPToolFunc - Callback function for MCP tools

   Synopsis:
   PetscErrorCode PetscMCPToolFunc(PetscMCP mcp, const char* arguments_json, char** result_json, void *ctx);

   Input Parameters:
+  mcp - the MCP context
.  arguments_json - JSON string containing tool arguments
-  ctx - user context

   Output Parameter:
.  result_json - JSON string containing the tool result (must be allocated with PetscMalloc)

   Level: developer
J*/
typedef PetscErrorCode (*PetscMCPToolFunc)(PetscMCP mcp, const char* arguments_json, char** result_json, void *ctx);

PETSC_EXTERN PetscErrorCode PetscMCPCreate(MPI_Comm comm, PetscMCP *mcp);
PETSC_EXTERN PetscErrorCode PetscMCPDestroy(PetscMCP *mcp);
PETSC_EXTERN PetscErrorCode PetscMCPRun(PetscMCP mcp);
PETSC_EXTERN PetscErrorCode PetscMCPRegisterTool(PetscMCP mcp, const char *name, const char *description, const char *input_schema_json, PetscMCPToolFunc func, void *ctx);
PETSC_EXTERN PetscErrorCode PetscMCPRegisterDefaultTools(PetscMCP mcp);

#endif
