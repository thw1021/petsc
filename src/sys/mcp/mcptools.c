#include <petscmcp.h>
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>
#include "../json/cJSON.h"

/* -------------------------------------------------------------------------- */
/*                               Built-in Tools                               */
/* -------------------------------------------------------------------------- */

static PetscErrorCode PetscMCPTool_GetVersion(PetscMCP mcp, const char* arguments_json, char** result_json, void *ctx)
{
  char version[256];
  cJSON *result;
  
  PetscFunctionBegin;
  PetscCall(PetscGetVersion(version, sizeof(version)));
  
  /* Return simple text content */
  /* Or return a JSON object if we want structured data. 
     The caller (PetscMCPHandleCallTool) expects a JSON string that it will parse 
     and put into the "content" field.
  */
  
  result = cJSON_CreateArray();
  cJSON *item = cJSON_CreateObject();
  cJSON_AddStringToObject(item, "type", "text");
  cJSON_AddStringToObject(item, "text", version);
  cJSON_AddItemToArray(result, item);
  
  *result_json = cJSON_PrintUnformatted(result);
  cJSON_Delete(result);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscMCPTool_OptionsView(PetscMCP mcp, const char* arguments_json, char** result_json, void *ctx)
{
  cJSON *result;
  PetscViewer viewer;
  FILE *fp;
  char buffer[8192];
  size_t nread;
  
  PetscFunctionBegin;
  /* PetscOptionsView requires an ASCII viewer. We'll use a temporary file. */
  
  fp = tmpfile();
  if (!fp) PetscFunctionReturn(PETSC_ERR_FILE_OPEN);
  
  PetscCall(PetscViewerASCIIOpenWithFILE(PETSC_COMM_SELF, fp, &viewer));
  PetscCall(PetscOptionsView(NULL, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  
  /* Read back the contents */
  rewind(fp);
  nread = fread(buffer, 1, sizeof(buffer) - 1, fp);
  buffer[nread] = '\0';
  fclose(fp);
  
  result = cJSON_CreateArray();
  cJSON *item = cJSON_CreateObject();
  cJSON_AddStringToObject(item, "type", "text");
  cJSON_AddStringToObject(item, "text", nread > 0 ? buffer : "No options set.");
  cJSON_AddItemToArray(result, item);
  
  *result_json = cJSON_PrintUnformatted(result);
  cJSON_Delete(result);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscMCPRegisterDefaultTools - Registers default PETSc tools

   Logically Collective

   Input Parameter:
.  mcp - the MCP context

   Level: developer
@*/
PetscErrorCode PetscMCPRegisterDefaultTools(PetscMCP mcp)
{
  PetscFunctionBegin;
  PetscCall(PetscMCPRegisterTool(mcp, "petsc_get_version", "Get PETSc version information", "{}", PetscMCPTool_GetVersion, NULL));
  PetscCall(PetscMCPRegisterTool(mcp, "petsc_options_view", "View the current PETSc options database", "{}", PetscMCPTool_OptionsView, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
