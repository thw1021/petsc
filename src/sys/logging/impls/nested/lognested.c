
#include <petscviewer.h>
#include "lognested.h"
#include "xmlviewer.h"

PETSC_INTERN PetscErrorCode PetscLogSetThreshold_Nested(PetscLogHandlerEntry h, PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;

  PetscFunctionBegin;
  if (oldThresh) *oldThresh = nested->threshold;
  if (newThresh == (PetscLogDouble)PETSC_DECIDE) newThresh = 0.01;
  if (newThresh == (PetscLogDouble)PETSC_DEFAULT) newThresh = 0.01;
  nested->threshold = PetscMax(newThresh, 0.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventGetNestedEvent(PetscLogHandlerEntry h, PetscLogRegistry registry, PetscLogEvent e, PetscLogEvent *nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  NestedIdPair           key;
  PetscHashIter          iter;
  PetscBool              missing;
  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromEvent(e);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    // register a new nested event
    char              name[BUFSIZ];
    PetscLogEventInfo event_info;
    PetscLogEventInfo nested_event_info;

    PetscCall(PetscLogRegistryEventGetInfo(registry, e, &event_info));
    PetscCall(PetscLogRegistryEventGetInfo(nested->state->registry, key.root, &nested_event_info));
    PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, event_info.name));
    PetscCall(PetscLogStateEventRegister(nested->state, name, 0, nested_event));
    PetscCall(PetscNestedHashIterSet(nested->pair_map, iter, *nested_event));
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, nested_event));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStageGetNestedEvent(PetscLogHandlerEntry h, PetscLogRegistry registry, PetscLogStage stage, PetscLogEvent *nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  NestedIdPair           key;
  PetscHashIter          iter;
  PetscBool              missing;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromStage(stage);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    PetscLogStageInfo stage_info;
    char              name[BUFSIZ];

    PetscCall(PetscLogRegistryStageGetInfo(registry, stage, &stage_info));
    if (key.root >= 0) {
      PetscLogEventInfo nested_event_info;

      PetscCall(PetscLogRegistryEventGetInfo(nested->state->registry, key.root, &nested_event_info));
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, stage_info.name));
    } else {
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s", stage_info.name));
    }
    PetscCall(PetscLogStateEventRegister(nested->state, name, nested->nested_stage_id, nested_event));
    PetscCall(PetscNestedHashIterSet(nested->pair_map, iter, *nested_event));
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, nested_event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCheckNested(PetscLogHandlerEntry h, NestedId leaf, PetscLogEvent nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  NestedIdPair           key;
  NestedId               val;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = leaf;
  PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
  PetscCheck(val == nested_event, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging events and stages are not nested, nested logging cannot be used");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventBegin_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscLogEvent e, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscLogEventGetNestedEvent(h, state->registry, e, &nested_event));
  PetscCall((*(nested->handler->event_begin))(nested->handler, nested->state, nested_event, t, o1, o2, o3, o4));
  PetscCall(PetscIntStackPush(nested->stack, nested_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventEnd_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscLogEvent e, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_event));
  if (PetscDefined(USE_DEBUG)) PetscCall(PetscLogNestedCheckNested(h, NestedIdFromEvent(e), nested_event));
  PetscCall((*(nested->handler->event_end))(nested->handler, nested->state, nested_event, t, o1, o2, o3, o4));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventSync_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscLogEvent e, MPI_Comm comm)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  if (!nested->handler->event_sync) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventGetNestedEvent(h, state->registry, e, &nested_event));
  PetscCall((*(nested->handler->event_sync))(nested->handler, nested->state, nested_event, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePush_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  if (nested->nested_stage_id == -1) PetscCall(PetscClassIdRegister("LogNestedStage", &nested->nested_stage_id));
  PetscCall(PetscLogStageGetNestedEvent(h, state->registry, stage, &nested_event));
  PetscCall((*(nested->handler->event_begin))(nested->handler, nested->state, nested_event, 0, NULL, NULL, NULL, NULL));
  PetscCall(PetscIntStackPush(nested->stack, nested_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePop_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_event));
  if (PetscDefined(USE_DEBUG)) PetscCall(PetscLogNestedCheckNested(h, NestedIdFromStage(stage), nested_event));
  PetscCall((*(nested->handler->event_end))(nested->handler, nested->state, nested_event, 0, NULL, NULL, NULL, NULL));
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
  nested->nested_stage_id = -1;
  nested->threshold       = 0.01;
  PetscCall(PetscNestedHashCreate(&nested->pair_map));
  PetscCall(PetscLogHandlerCreate_Default(&nested->handler));
  PetscCall(PetscLogStateStageRegister(nested->state, "", &root_stage));
  PetscAssert(root_stage == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "root stage not zero");
  PetscCall((*(nested->handler->impl->stage_push))(nested->handler, nested->state, root_stage));
  PetscCall(PetscLogStateStagePush(nested->state, root_stage));
  PetscCall(PetscIntStackPush(nested->stack, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjectCreate_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscObject obj)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;

  PetscFunctionBegin;
  if (nested->handler->object_create) PetscCall((*(nested->handler->object_create))(nested->handler, state, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjectDestroy_Nested(PetscLogHandlerEntry h, PetscLogState state, PetscObject obj)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;

  PetscFunctionBegin;
  if (nested->handler->object_destroy) PetscCall((*(nested->handler->object_destroy))(nested->handler, state, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Nested(PetscLogHandlerEntry h)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->impl->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogStateStagePop(nested->state));
  PetscCall((*(nested->handler->impl->stage_pop))(nested->handler, nested->state, 0));
  PetscCall(PetscLogStateDestroy(nested->state));
  PetscCall(PetscIntStackDestroy(nested->stack));
  PetscCall(PetscNestedHashDestroy(&nested->pair_map));
  PetscCall(PetscLogHandlerEntryDestroy(&nested->handler));
  PetscCall(PetscFree(nested));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedEventNodesOrderDepthFirst(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscInt *num_descendants)
{
  PetscInt node, start_loc;
  PetscFunctionBegin;

  node      = 0;
  start_loc = 0;
  while (node < num_nodes) {
    if (tree[node].parent == parent) {
      PetscInt             num_this_descendants = 0;
      PetscNestedEventNode tmp                  = tree[start_loc];
      tree[start_loc]                           = tree[node];
      tree[node]                                = tmp;
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
  PetscMPIInt           size;
  PetscInt              num_nodes;
  PetscInt              num_map_entries;
  PetscEventPerfInfo   *perf;
  NestedIdPair         *keys;
  NestedId             *vals;
  PetscInt              offset;
  PetscInt              num_descendants;
  PetscNestedEventNode *tree;

  PetscFunctionBegin;
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &num_nodes));
  PetscCall(PetscCalloc1(num_nodes, &tree));
  for (PetscInt node = 0; node < num_nodes; node++) {
    tree[node].id = node;
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, node, &tree[node].name));
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

    PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, leaf_local, &leaf_global));
    if (root_local >= 0) {
      PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, root_local, &root_global));
      tree[leaf_global].parent = root_global;
    }
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
    } else {
      PetscCall(PetscArrayzero(&perf[node], 1));
    }
  }
  *tree_p = tree;
  *perf_p = perf;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscLogHandlerEntry handler, PetscViewer viewer)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)handler->impl->ctx;
  PetscNestedEventNode  *nodes;
  PetscEventPerfInfo    *perf;
  PetscLogGlobalNames    global_events;
  PetscNestedEventTree   tree;
  PetscViewerFormat      format;
  MPI_Comm               comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, nested->state->registry, &global_events));
  PetscCall(PetscLogNestedCreatePerfNodes(comm, nested, global_events, &nodes, &perf));
  tree.comm          = comm;
  tree.global_events = global_events;
  tree.perf          = perf;
  tree.nodes         = nodes;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_XML) {
    PetscCall(PetscLogView_Nested_XML(nested, &tree, viewer));
  } else if (format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    PetscCall(PetscLogView_Nested_Flamegraph(nested, &tree, viewer));
  } else SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "No nested viewer for this format");
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscCall(PetscFree(tree.nodes));
  PetscCall(PetscFree(tree.perf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Nested(PetscLogHandlerEntry *handler_p)
{
  PetscLogHandlerEntry handler;

  PetscFunctionBegin;
  PetscCall(PetscNew(handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&handler->impl));
  PetscCall(PetscLogHandlerContextCreate_Nested((PetscLogHandler_Nested *)&handler->impl->ctx));
  handler->impl->type       = PETSC_LOG_HANDLER_NESTED;
  handler->impl->view       = PetscLogView_Nested;
  handler->impl->destroy    = PetscLogHandlerDestroy_Nested;
  handler->impl->stage_push = PetscLogStagePush_Nested;
  handler->impl->stage_pop  = PetscLogStagePop_Nested;
  handler->event_begin      = PetscLogEventBegin_Nested;
  handler->event_end        = PetscLogEventEnd_Nested;
  handler->event_sync       = PetscLogEventSync_Nested;
  handler->object_create    = PetscLogObjectCreate_Nested;
  handler->object_destroy   = PetscLogObjectDestroy_Nested;
  PetscFunctionReturn(PETSC_SUCCESS);
}
