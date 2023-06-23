
#include <petscviewer.h>
#include "lognested.h"
#include "xmlviewer.h"

static PetscErrorCode PetscLogEventBegin_Nested(PetscLogHandler h, PetscLogState state, PetscLogEvent e, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) h->impl->ctx;
  NestedIdPair key;
  NestedId     nested_id;
  PetscHashIter iter;
  PetscLogEvent nested_event;
  PetscBool missing;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromEvent(e);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    // register a new nested event
    char name[BUFSIZ];
    PetscEventRegInfo event_info;
    PetscEventRegInfo nested_event_info;

    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    PetscCall(PetscLogRegistryEventGetInfo(nested->state->registry, key.root, &nested_event_info));
    PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, event_info.name));
    PetscCall(PetscLogStateEventRegister(nested->state, name, 0, &nested_event));
    nested_id = NestedIdFromEvent(nested_event);
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, &nested_id));
    nested_event = NestedIdToEvent(nested_id);
  }
  PetscCall(nested->handler->event_begin(nested->handler, nested->state, nested_event, t, o1, o2, o3, o4));
  PetscCall(PetscIntStackPush(nested->stack, nested_id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventEnd_Nested(PetscLogHandler h, PetscLogState state, PetscLogEvent e, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) h->impl->ctx;
  NestedId nested_id;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_id));
  if (PetscDefined(USE_DEBUG)) {
    NestedIdPair key;
    NestedId     val;

    PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
    key.leaf = NestedIdFromEvent(e);
    PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
    PetscCheck(val == nested_id, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging events appear to be unnested, nested logging cannot be used");
  }
  PetscCall(nested->handler->event_end(nested->handler, nested->state, NestedIdToEvent(nested_id), t, o1, o2, o3, o4));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePush_Nested(PetscLogHandler h, PetscLogState state, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) h->impl->ctx;
  NestedIdPair key;
  NestedId     nested_id;
  PetscHashIter iter;
  PetscBool missing;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromStage(stage);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    PetscLogEvent e;
    PetscStageRegInfo stage_info;
    char name[BUFSIZ];

    PetscCall(PetscLogRegistryStageGetInfo(nested->state->registry, stage, &stage_info));
    if (key.root >= 0) {
      PetscEventRegInfo nested_event_info;

      PetscCall(PetscLogRegistryEventGetInfo(nested->state->registry, key.root, &nested_event_info));
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, stage_info.name));
    } else {
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s", stage_info.name));
    }
    PetscCall(PetscLogStateEventRegister(nested->state, name, nested->nested_stage_id, &e));
    nested_id = NestedIdFromEvent(e);
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, &nested_id));
  }
  PetscCall(nested->handler->event_begin(nested->handler, nested->state, NestedIdToEvent(nested_id), 0, NULL, NULL, NULL, NULL));
  PetscCall(PetscIntStackPush(nested->stack, nested_id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePop_Nested(PetscLogHandler h, PetscLogState state, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) h->impl->ctx;
  NestedId nested_id;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_id));
  if (PetscDefined(USE_DEBUG)) {
    NestedIdPair key;
    NestedId     val;

    PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
    key.leaf = NestedIdFromStage(stage);
    PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
    PetscCheck(val == nested_id, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging stages appear to be unnested, nested logging cannot be used");
  }
  PetscCall(nested->handler->event_end(nested->handler, nested->state, NestedIdToEvent(nested_id), 0, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerContextCreate_Nested(PetscLogHandler_Nested *nested_p)
{
  PetscLogStage          root_stage;
  PetscLogHandler_Nested nested;

  PetscFunctionBegin;
  PetscCall(PetscNew(nested_p));
  nested = *nested_p;
  PetscCall(PetscLogStateCreate(&nested->state));
  PetscCall(PetscIntStackCreate(&nested->stack));
  PetscCall(PetscClassIdRegister("LogNestedStage", &nested->nested_stage_id));
  PetscCall(PetscNestedHashCreate(&nested->pair_map));
  PetscCall(PetscLogHandlerCreate_Default(&nested->handler));
  PetscCall(PetscLogStateStageRegister(nested->state, "", &root_stage));
  PetscAssert(root_stage == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "root stage not zero");
  PetscCall((*(nested->handler->impl->stage_push))(nested->handler, nested->state, root_stage));
  PetscCall(PetscLogStateStagePush(nested->state, root_stage));
  PetscCall(PetscIntStackPush(nested->stack, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerContextDestroy_Nested(PetscLogHandler h)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) h->impl->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogStateStagePop(nested->state));
  PetscCall((*(nested->handler->impl->stage_pop))(nested->handler, nested->state, 0));
  PetscCall(PetscLogStateDestroy(nested->state));
  PetscCall(PetscIntStackDestroy(nested->stack));
  PetscCall(PetscNestedHashDestroy(&nested->pair_map));
  PetscCall(PetscLogHandlerDestroy(&nested->handler));
  PetscCall(PetscFree(nested));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerCreate_Nested(PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscNew(handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&handler->impl));
  handler->impl->type = PETSC_LOG_HANDLER_NESTED;
  PetscCall(PetscLogHandlerContextCreate_Nested((PetscLogHandler_Nested *) &handler->impl->ctx));
  handler->impl->destroy = PetscLogHandlerContextDestroy_Nested;
  handler->event_begin = PetscLogEventBegin_Nested;
  handler->event_end = PetscLogEventEnd_Nested;
  handler->impl->stage_push = PetscLogStagePush_Nested;
  handler->impl->stage_pop = PetscLogStagePop_Nested;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogNestedBegin(void)
{
  int i_free = -1;
  
  PetscFunctionBegin;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i];
    if (h) {
     if (h->impl->type == PETSC_LOG_HANDLER_NESTED) PetscFunctionReturn(PETSC_SUCCESS);
    } else if (i_free < 0) {
      i_free = i;
    }
  }
  PetscCheck(i_free >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Too many log handlers, cannot created nested handler");
  PetscCall(PetscLogHandlerCreate_Nested(&PetscLogHandlers[i_free]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedEventNodesOrderDepthFirst(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscInt *num_descendants)
{
  PetscInt node, start_loc;
  PetscFunctionBegin;

  node = 0;
  start_loc = 0;
  while (node < num_nodes) {
    if (tree[node].parent == parent) {
      PetscInt num_this_descendants = 0;
      PetscNestedEventNode tmp = tree[start_loc];
      tree[start_loc] = tree[node];
      tree[node] = tmp;
      PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes - start_loc - 1, tree[start_loc].id, &tree[start_loc + 1], &num_this_descendants));
      tree[start_loc].num_descendants = num_this_descendants;
      *num_descendants += 1 + num_this_descendants;
      start_loc += 1 + num_this_descendants;
      node = start_loc;
    } else {
      node++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCreatePerfNodes(MPI_Comm comm, PetscLogHandler_Nested nested, PetscLogGlobalNames global_events, PetscNestedEventNode **tree_p, PetscEventPerfInfo **perf_p)
{
  PetscMPIInt size;
  PetscInt num_nodes;
  PetscInt num_map_entries;
  PetscEventPerfInfo *perf;
  NestedIdPair *keys;
  NestedId    *vals;
  PetscInt     offset;
  PetscInt num_descendants;
  PetscNestedEventNode *tree;

  PetscFunctionBegin;
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &num_nodes));
  PetscCall(PetscCalloc1(num_nodes, &tree));
  for (PetscInt node = 0; node < num_nodes; node++) {
    tree[node].id = node;
    tree[node].parent = -1;
  }
  PetscCall(PetscNestedHashGetSize(nested->pair_map, &num_map_entries));
  PetscCall(PetscMalloc2(num_map_entries, &keys, num_map_entries, &vals));
  offset = 0;
  PetscCall(PetscNestedHashGetPairs(nested->pair_map, &offset, keys, vals));
  for (PetscInt k = 0; k < num_map_entries; k++) {
    NestedId root_local = keys[k].root;
    NestedId leaf_local = vals[k];
    PetscInt root_global;
    PetscInt leaf_global;

    PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, root_local, &root_global));
    PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, leaf_local, &leaf_global));
    tree[leaf_global].parent = root_global;
  }
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) { // get missing parents from other processes
    PetscInt *parents;

    PetscCall(PetscMalloc1(num_nodes, &parents));
    for (PetscInt node = 0; node < num_nodes; node++) parents[node] = tree[node].parent;
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, parents, num_nodes, MPIU_INT, MPI_MAX, comm));
    for (PetscInt node = 0; node < num_nodes; node++) tree[node].parent = parents[node];
    PetscCall(PetscFree(parents));
  }

  num_descendants = 0;
  PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes, -1, tree, &num_descendants));
  PetscAssert(num_descendants == num_nodes, comm, PETSC_ERR_PLIB, "Failed tree ordering invariant");

  PetscCall(PetscCalloc1(num_nodes, &perf));
  for (PetscInt node = 0; node < num_nodes; node++) {
    PetscInt event_id;

    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, node, &event_id));
    if (event_id >= 0) {
      PetscEventPerfInfo *event_info;

      PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(nested->handler, 0, event_id, &event_info));
      perf[node] = *event_info;
    }
  }
  *tree_p = tree;
  *perf_p = perf;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) handler->impl->ctx;
  PetscNestedEventNode *nodes;
  PetscEventPerfInfo *perf;
  PetscLogGlobalNames global_events;
  PetscNestedEventTree tree;
  PetscViewerFormat format;
  MPI_Comm comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, nested->state->registry, &global_events));
  PetscCall(PetscLogNestedCreatePerfNodes(comm, nested, global_events, &nodes, &perf));
  tree.comm = comm;
  tree.global_events = global_events;
  tree.perf = perf;
  tree.nodes = nodes;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_XML) {
    PetscCall(PetscLogView_Nested_XML(nested, &tree, viewer));
  } else if (format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    PetscCall(PetscLogView_Nested_Flamegraph(nested, &tree, viewer));
  } else SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "No nested viewer for this format");
  PetscFunctionReturn(PETSC_SUCCESS);
}

