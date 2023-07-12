const char help[] = "Test PetscLogEventsPause() and PetscLogEventsUnpause()";

#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscLogStage main_stage, unrelated_stage;
  PetscLogEvent runtime_event, unrelated_event;
  PetscClassId  runtime_classid, unrelated_classid;
  PetscBool     main_visible      = PETSC_FALSE;
  PetscBool     unrelated_visible = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, NULL);
  PetscCall(PetscOptionsBool("-main_visible", "The logging visibility of the main stage", NULL, main_visible, &main_visible, NULL));
  PetscCall(PetscOptionsBool("-unrelated_visible", "The logging visibility of the unrelated stage", NULL, unrelated_visible, &unrelated_visible, NULL));
  PetscOptionsEnd();

  // This test simulates a program with unrelated logging stages and events
  // that has to "stop the world" to lazily initialize a runtime.
  //
  // - Pausing events should send the log data for the runtime initalization
  //   to the Main Stage
  //
  // - Turning the Main Stage invisible should hide it from -log_view
  //
  // So the runtime intialization should be more or less missing from -log_view.

  PetscCall(PetscClassIdRegister("External runtime", &runtime_classid));
  PetscCall(PetscLogEventRegister("External runtime initialization", runtime_classid, &runtime_event));

  PetscCall(PetscClassIdRegister("Unrelated class", &unrelated_classid));
  PetscCall(PetscLogEventRegister("Unrelated event", unrelated_classid, &unrelated_event));
  PetscCall(PetscLogStageRegister("Unrelated stage", &unrelated_stage));
  PetscCall(PetscLogStageGetId("Main Stage", &main_stage));
  PetscCall(PetscLogStageSetVisible(main_stage, main_visible));
  PetscCall(PetscLogStageSetVisible(unrelated_stage, unrelated_visible));

  PetscCall(PetscLogStagePush(unrelated_stage));
  PetscCall(PetscLogEventBegin(unrelated_event, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventsPause());
  PetscCall(PetscLogEventBegin(runtime_event, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.2));
  PetscCall(PetscLogEventEnd(runtime_event, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventsResume());

  PetscCall(PetscLogEventEnd(unrelated_event, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # main stage invisible, "External runtime initialization" shouldn't appear in the log
  test:
    requires: defined(PETSC_USE_LOG)
    suffix: 0
    args: -log_view -unrelated_visible
    filter: grep -o "\\(External runtime initialization\\|Unrelated event\\)"

  # unrelated stage invisible, "Unrelated event" shouldn't appear in the log
  test:
    requires: defined(PETSC_USE_LOG)
    suffix: 1
    args: -log_view -main_visible
    filter: grep -o "\\(External runtime initialization\\|Unrelated event\\)"

TEST*/
