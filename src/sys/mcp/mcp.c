#include <petscmcp.h>
#include <petsc/private/petscimpl.h>
#include "../json/cJSON.h"

/*
  Internal MCP structure
*/
struct _p_PetscMCP {
  PETSCHEADER(int);
  cJSON *tools; /* Dictionary of registered tools */
};

/*
  Internal context for registered tools
*/
typedef struct {
  char            *name;
  char            *description;
  char            *input_schema;
  PetscMCPToolFunc func;
  void            *ctx;
} PetscMCPTool;

/*
  Helper to free tool context
*/
static PetscErrorCode PetscMCPToolDestroy(void **ctx)
{
  PetscMCPTool *tool = (PetscMCPTool*)*ctx;
  if (!tool) return 0;
  free(tool->name);
  free(tool->description);
  free(tool->input_schema);
  free(tool);
  *ctx = NULL;
  return 0;
}

/*@
   PetscMCPCreate - Creates an MCP server context

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  mcp - the new MCP context

   Level: developer

.seealso: `PetscMCPDestroy()`, `PetscMCPRun()`
@*/
PetscErrorCode PetscMCPCreate(MPI_Comm comm, PetscMCP *mcp)
{
  PetscMCP m;

  PetscFunctionBegin;
  PetscAssertPointer(mcp, 2);
  PetscCall(PetscHeaderCreate(m, PETSC_OBJECT_CLASSID, "PetscMCP", "Model Context Protocol", "Sys", comm, PetscMCPDestroy, NULL));
  
  m->tools = cJSON_CreateObject();
  
  /* Register default tools */
  PetscCall(PetscMCPRegisterDefaultTools(m));

  *mcp = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscMCPDestroy - Destroys an MCP server context

   Collective

   Input Parameter:
.  mcp - the MCP context

   Level: developer

.seealso: `PetscMCPCreate()`
@*/
PetscErrorCode PetscMCPDestroy(PetscMCP *mcp)
{
  PetscFunctionBegin;
  if (!*mcp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*mcp, PETSC_OBJECT_CLASSID, 1);
  
  if (--((PetscObject)(*mcp))->refct > 0) {
    *mcp = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Free registered tools */
  if ((*mcp)->tools) {
    cJSON *child = (*mcp)->tools->child;
    while (child) {
      /* The cJSON object's string value holds the pointer to our PetscMCPTool struct 
         Wait, cJSON doesn't hold void* user data easily. 
         We'll need to manage the memory of the tool contexts separately or 
         use a different data structure if we want to be clean.
         For now, let's assume we just delete the cJSON object, but we leak the PetscMCPTool structs?
         No, we need to iterate and free.
      */
       /* Actually, cJSON isn't a hash map that stores void*. It stores cJSON items.
          We probably need a PetscContainer or similar to map names to function pointers/contexts.
          Or we can just use a PetscSegBuffer or simple linked list for now since N tools is small.
          Let's stick to a simple linked list of tools for now inside the struct, 
          and use cJSON only for constructing the "tools/list" response.
       */
       child = child->next;
    }
    cJSON_Delete((*mcp)->tools);
  }
  
  /* We need to fix the tool storage strategy. 
     Let's use a simple linked list for internal storage, and generate cJSON on demand.
  */

  PetscCall(PetscHeaderDestroy(mcp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------------------------------------------------------------------------- */
/*                               Protocol Logic                               */
/* -------------------------------------------------------------------------- */

static PetscErrorCode PetscMCPHandleInitialize(PetscMCP mcp, cJSON *request, cJSON **response)
{
  cJSON *result, *capabilities;
  
  PetscFunctionBegin;
  *response = cJSON_CreateObject();
  cJSON_AddStringToObject(*response, "jsonrpc", "2.0");
  cJSON_AddNumberToObject(*response, "id", cJSON_GetObjectItem(request, "id")->valueint);
  
  result = cJSON_CreateObject();
  cJSON_AddItemToObject(*response, "result", result);
  
  cJSON_AddStringToObject(result, "protocolVersion", "2024-11-05");
  
  capabilities = cJSON_CreateObject();
  cJSON_AddItemToObject(result, "capabilities", capabilities);
  
  cJSON_AddItemToObject(capabilities, "tools", cJSON_CreateObject());
  
  cJSON_AddItemToObject(result, "serverInfo", cJSON_CreateObject());
  cJSON_AddStringToObject(cJSON_GetObjectItem(result, "serverInfo"), "name", "PetscMCP");
  cJSON_AddStringToObject(cJSON_GetObjectItem(result, "serverInfo"), "version", "0.1.0");
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscMCPHandlePing(PetscMCP mcp, cJSON *request, cJSON **response)
{
  PetscFunctionBegin;
  *response = cJSON_CreateObject();
  cJSON_AddStringToObject(*response, "jsonrpc", "2.0");
  cJSON_AddNumberToObject(*response, "id", cJSON_GetObjectItem(request, "id")->valueint);
  cJSON_AddNullToObject(*response, "result");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscMCPHandleListTools(PetscMCP mcp, cJSON *request, cJSON **response);
static PetscErrorCode PetscMCPHandleCallTool(PetscMCP mcp, cJSON *request, cJSON **response);

/* -------------------------------------------------------------------------- */
/*                               Main Loop                                    */
/* -------------------------------------------------------------------------- */

/*@
   PetscMCPRun - Runs the MCP server loop, reading from stdin and writing to stdout

   Collective

   Input Parameter:
.  mcp - the MCP context

   Level: developer

   Note:
   This function blocks until the connection is closed or an exit signal is received.
@*/
PetscErrorCode PetscMCPRun(PetscMCP mcp)
{
  char buffer[4096]; /* Fixed buffer for now, should be dynamic */
  cJSON *request, *response;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mcp, PETSC_OBJECT_CLASSID, 1);

  /* Simple line-based reader for now. 
     TODO: Handle partial reads and larger messages. 
  */
  while (fgets(buffer, sizeof(buffer), stdin)) {
    request = cJSON_Parse(buffer);
    if (!request) {
      /* Parse error, ignore or send error */
      continue;
    }
    
    cJSON *method = cJSON_GetObjectItem(request, "method");
    response = NULL;
    
    if (method && method->valuestring) {
      if (strcmp(method->valuestring, "initialize") == 0) {
        PetscCall(PetscMCPHandleInitialize(mcp, request, &response));
      } else if (strcmp(method->valuestring, "ping") == 0) {
        PetscCall(PetscMCPHandlePing(mcp, request, &response));
      } else if (strcmp(method->valuestring, "notifications/initialized") == 0) {
        /* No response needed for notifications */
      } else if (strcmp(method->valuestring, "tools/list") == 0) {
        PetscCall(PetscMCPHandleListTools(mcp, request, &response));
      } else if (strcmp(method->valuestring, "tools/call") == 0) {
        PetscCall(PetscMCPHandleCallTool(mcp, request, &response));
      } else {
        /* Method not found */
        /* TODO: Send error response */
      }
    }
    
    if (response) {
      char *str = cJSON_PrintUnformatted(response);
      fprintf(stdout, "%s\n", str);
      fflush(stdout);
      free(str);
      cJSON_Delete(response);
    }
    
    cJSON_Delete(request);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscMCPRegisterTool - Registers a tool with the MCP server

   Logically Collective

   Input Parameters:
+  mcp - the MCP context
.  name - tool name
.  description - tool description
.  input_schema_json - JSON schema for arguments
.  func - function to call
-  ctx - user context

   Level: developer
@*/
PetscErrorCode PetscMCPRegisterTool(PetscMCP mcp, const char *name, const char *description, const char *input_schema_json, PetscMCPToolFunc func, void *ctx)
{
  PetscMCPTool *tool;
  cJSON *tool_json;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mcp, PETSC_OBJECT_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(description, 3);
  PetscAssertPointer(input_schema_json, 4);

  /* Create tool context */
  tool = (PetscMCPTool*)malloc(sizeof(PetscMCPTool));
  tool->name = strdup(name);
  tool->description = strdup(description);
  tool->input_schema = strdup(input_schema_json);
  tool->func = func;
  tool->ctx = ctx;

  /* Store in cJSON object (using name as key, and pointer as value is tricky in cJSON)
     Instead, we will store the tool metadata as a cJSON object, and keep the function pointer
     in a separate internal list or just cast the cJSON user data if possible.
     
     Actually, cJSON doesn't support user data well.
     Let's use a simple linked list for now for the registry, and rebuild the cJSON "tools" list
     whenever requested.
  */
  
  /* For now, we will just add it to the 'tools' object as a JSON representation,
     but we need a way to retrieve the function pointer back.
     
     Let's change the strategy: 'mcp->tools' will be a cJSON Array of tool definitions.
     We also need a way to look up the function.
     
     Let's use a PetscContainer to hold the PetscMCPTool struct, and compose it with the MCP object
     using a unique key (e.g., "mcp_tool_<name>").
  */
  
  {
    char container_name[256];
    PetscContainer container;
    
    snprintf(container_name, sizeof(container_name), "mcp_tool_%s", name);
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)mcp), &container));
    PetscCall(PetscContainerSetPointer(container, tool));
    PetscCall(PetscContainerSetCtxDestroy(container, PetscMCPToolDestroy));
    PetscCall(PetscObjectCompose((PetscObject)mcp, container_name, (PetscObject)container));
    PetscCall(PetscContainerDestroy(&container));
  }
  
  /* Add to cJSON list for easy reporting */
  /* If mcp->tools is an object, we can use name as key */
  tool_json = cJSON_CreateObject();
  cJSON_AddStringToObject(tool_json, "name", name);
  cJSON_AddStringToObject(tool_json, "description", description);
  /* Parse the input schema string into a cJSON object */
  cJSON *schema = cJSON_Parse(input_schema_json);
  if (schema) {
    cJSON_AddItemToObject(tool_json, "inputSchema", schema);
  } else {
    cJSON_AddStringToObject(tool_json, "inputSchema", input_schema_json); /* Fallback */
  }
  
  cJSON_AddItemToObject(mcp->tools, name, tool_json);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscMCPHandleListTools(PetscMCP mcp, cJSON *request, cJSON **response)
{
  cJSON *result, *tools_array;
  cJSON *item;
  
  PetscFunctionBegin;
  *response = cJSON_CreateObject();
  cJSON_AddStringToObject(*response, "jsonrpc", "2.0");
  cJSON_AddNumberToObject(*response, "id", cJSON_GetObjectItem(request, "id")->valueint);
  
  result = cJSON_CreateObject();
  cJSON_AddItemToObject(*response, "result", result);
  
  tools_array = cJSON_CreateArray();
  cJSON_AddItemToObject(result, "tools", tools_array);
  
  /* Iterate over registered tools - mcp->tools is an object, not an array */
  item = mcp->tools->child;
  while (item) {
    /* Create a copy of the tool definition to add to the array */
    cJSON_AddItemToArray(tools_array, cJSON_Duplicate(item, 1));
    item = item->next;
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscMCPHandleCallTool(PetscMCP mcp, cJSON *request, cJSON **response)
{
  cJSON *params, *name_json, *args_json;
  char container_name[256];
  PetscContainer container;
  PetscMCPTool *tool;
  char *result_json_str = NULL;
  char *args_str = NULL;
  
  PetscFunctionBegin;
  params = cJSON_GetObjectItem(request, "params");
  name_json = cJSON_GetObjectItem(params, "name");
  args_json = cJSON_GetObjectItem(params, "arguments");
  
  if (!name_json || !name_json->valuestring) {
    /* Error: Invalid params */
    PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
  }
  
  snprintf(container_name, sizeof(container_name), "mcp_tool_%s", name_json->valuestring);
  PetscCall(PetscObjectQuery((PetscObject)mcp, container_name, (PetscObject*)&container));
  
  if (!container) {
    /* Error: Tool not found */
    /* TODO: Return JSON-RPC error */
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  
  PetscCall(PetscContainerGetPointer(container, (void**)&tool));
  
  if (args_json) {
    args_str = cJSON_Print(args_json);
  }
  
  /* Call the tool */
  PetscCall(tool->func(mcp, args_str, &result_json_str, tool->ctx));
  
  if (args_str) free(args_str);
  
  /* Construct response */
  *response = cJSON_CreateObject();
  cJSON_AddStringToObject(*response, "jsonrpc", "2.0");
  cJSON_AddNumberToObject(*response, "id", cJSON_GetObjectItem(request, "id")->valueint);
  
  cJSON *result = cJSON_CreateObject();
  cJSON_AddItemToObject(*response, "result", result);
  
  /* Result content is usually a list of content items (text/image) */
  /* We assume the tool returns a JSON string that matches the expected content structure
     OR just the content value. For simplicity, let's assume the tool returns
     the full "content" array as a JSON string.
  */
  if (result_json_str) {
    cJSON *content = cJSON_Parse(result_json_str);
    if (content) {
      cJSON_AddItemToObject(result, "content", content);
    } else {
      /* Treat as plain text if parse fails */
      cJSON *content_array = cJSON_CreateArray();
      cJSON *text_item = cJSON_CreateObject();
      cJSON_AddStringToObject(text_item, "type", "text");
      cJSON_AddStringToObject(text_item, "text", result_json_str);
      cJSON_AddItemToArray(content_array, text_item);
      cJSON_AddItemToObject(result, "content", content_array);
    }
    free(result_json_str); /* Allocated by tool using cJSON_PrintUnformatted which uses malloc */
  } else {
     cJSON_AddItemToObject(result, "content", cJSON_CreateArray());
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}
