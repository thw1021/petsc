
static char help[] = "Tests nested events.\n\n";

#include <petscsys.h>

static PetscErrorCode CallEvents(PetscLogEvent event1, PetscLogEvent event2, PetscLogEvent event3)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(event1, 0, 0, 0, 0));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(event2, 0, 0, 0, 0));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(event3, 0, 0, 0, 0));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventEnd(event3, 0, 0, 0, 0));
  PetscCall(PetscLogEventEnd(event2, 0, 0, 0, 0));
  PetscCall(PetscLogEventEnd(event1, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscLogStage stage1, stage2;
  PetscLogEvent event1, event2, event3;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscLogEventRegister("Event2", 0, &event2));
  PetscCall(PetscLogEventRegister("Event1", 0, &event1));
  PetscCall(PetscLogEventRegister("Event3", 0, &event3));
  PetscCall(PetscLogStageRegister("Stage1", &stage1));
  PetscCall(PetscLogStageRegister("Stage2", &stage2));

  PetscCall(CallEvents(event1, event2, event3));

  PetscCall(PetscLogStagePush(stage1));
  {
    PetscCall(CallEvents(event1, event2, event3));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage2));
  {
    PetscCall(CallEvents(event1, event2, event3));

    PetscCall(PetscLogStagePush(stage1));
    {
      PetscCall(CallEvents(event1, event2, event3));
    }
    PetscCall(PetscLogStagePop());

    PetscCall(PetscLogEventBegin(event1, 0, 0, 0, 0));
    {
      PetscCall(PetscLogStagePush(stage1));
      {
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

 TEST*/
