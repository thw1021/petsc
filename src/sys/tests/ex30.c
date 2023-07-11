
static char help[] = "Tests nested events.\n\n";

#include <petscsys.h>

static PetscErrorCode CallEvents(PetscLogEvent event1, PetscLogEvent event2, PetscLogEvent event3)
{
  char *data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(event1, 0, 0, 0, 0));
  PetscCall(PetscSleep(0.05));
  PetscCall(PetscLogEventBegin(event2, 0, 0, 0, 0));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(event3, 0, 0, 0, 0));
  PetscCall(PetscCalloc1(1048576, &data));
  PetscCall(PetscFree(data));
  PetscCall(PetscSleep(0.15));
  PetscCall(PetscLogEventEnd(event3, 0, 0, 0, 0));
  PetscCall(PetscLogEventEnd(event2, 0, 0, 0, 0));
  PetscCall(PetscLogEventEnd(event1, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscLogStage  stage1, stage2;
  PetscLogEvent  event1, event2, event3;
  PetscMPIInt    rank;
  PetscContainer container;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) {
    PetscCall(PetscLogEventRegister("Event3", 0, &event3));
    PetscCall(PetscLogEventRegister("Event2", 0, &event2));
    PetscCall(PetscLogEventRegister("Event1", 0, &event1));
    PetscCall(PetscLogStageRegister("Stage2", &stage2));
    PetscCall(PetscLogStageRegister("Stage1", &stage1));
  } else {
    PetscCall(PetscLogEventRegister("Event2", 0, &event2));
    PetscCall(PetscLogEventRegister("Event1", 0, &event1));
    PetscCall(PetscLogEventRegister("Event3", 0, &event3));
    PetscCall(PetscLogStageRegister("Stage1", &stage1));
    PetscCall(PetscLogStageRegister("Stage2", &stage2));
  }

  PetscCall(CallEvents(event1, event2, event3));

  PetscCall(PetscLogStagePush(stage1));
  {
    PetscCall(PetscSleep(0.1));
    PetscCall(CallEvents(event1, event2, event3));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &container));
  PetscCall(PetscContainerDestroy(&container));

  PetscCall(PetscLogStagePush(stage2));
  {
    PetscCall(PetscSleep(0.1));
    PetscCall(CallEvents(event1, event2, event3));

    PetscCall(PetscLogStagePush(stage1));
    {
      PetscCall(PetscSleep(0.1));
      PetscCall(CallEvents(event1, event2, event3));
    }
    PetscCall(PetscLogStagePop());

    PetscCall(PetscLogEventBegin(event1, 0, 0, 0, 0));
    {
      PetscCall(PetscSleep(0.1));
      PetscCall(PetscLogStagePush(stage1));
      {
        PetscCall(PetscSleep(0.1));
        PetscCall(CallEvents(event1, event2, event3));
      }
      PetscCall(PetscLogStagePop());
    }
    PetscCall(PetscLogEventEnd(event1, 0, 0, 0, 0));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2}}

  test:
    suffix: 1
    nsize: {{1 2}}
    requires: defined(PETSC_USE_LOG)
    args: -log_view ::ascii_flamegraph
    filter: sed -E "s/ [0-9]+/ time_removed/g"

  test:
    suffix: 2
    requires: defined(PETSC_USE_LOG)
    nsize: 1
    args: -log_trace

  test:
    suffix: 3
    nsize: 1
    requires: defined(PETSC_USE_LOG)
    args: -log_include_actions -log_include_objects -log_all
    temporaries: Log.0
    filter: cat Log.0 | grep "\\(Actions accomplished\\|Objects created\\)"

  test:
    suffix: 4
    nsize: 1
    requires: defined(PETSC_USE_LOG)
    args: -log_view ::ascii_csv
    filter: grep "Event[123]" | grep -v "PCMPI"

  test:
    suffix: 5
    nsize: 1
    requires: defined(PETSC_USE_LOG) defined(PETSC_HAVE_MPE)
    args: -log_mpe ex30_mpe
    temporaries: ex30_mpe.clog2
    filter: strings ex30_mpe.clog2 | grep "Event[123]"

 TEST*/
