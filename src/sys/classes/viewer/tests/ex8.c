static char help[] = "Tests CGNS viewers.\n\n";

#include <petscsys.h>
#include <petscviewer.h>
#include <petscdm.h>

static PetscErrorCode TestOpen(PetscFileMode mode, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCGNSOpen(PETSC_COMM_WORLD, "cgns.cgns", mode, viewer));
  PetscCall(PetscViewerSetUp(*viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetLongDescription(PetscInt *len, char **desc)
{
  const PetscInt desc_len = 4096, desc_dup_len = 32;
  const char     desc_dup[] = "this is a far longer description";

  PetscFunctionBegin;
  *len = desc_len;
  PetscCall(PetscCalloc1(desc_len + 1, desc));
  for (PetscInt i = 0; i < desc_len; i += desc_dup_len) {
    PetscCall(PetscMemcpy(&(*desc)[i], desc_dup, desc_dup_len));
  }
  (*desc)[desc_len] = '\0';
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestWriteDescriptors(PetscViewer viewer)
{
  char    *desc;
  PetscInt desc_len;

  PetscFunctionBegin;
  PetscCall(PetscViewerCGNSSetDescriptor(viewer, "Help", help));

  PetscCall(GetLongDescription(&desc_len, &desc));
  PetscCall(PetscViewerCGNSSetDescriptor(viewer, "Long Description", desc));
  PetscCall(PetscFree(desc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestReadDescriptors(PetscViewer viewer)
{
  char    **descriptors, **names, *expected_desc;
  PetscInt  num_descriptors, expected_desc_len;
  PetscBool is_same;

  PetscFunctionBegin;
  PetscCall(PetscViewerCGNSGetDescriptors(viewer, &num_descriptors, &names, &descriptors));
  PetscCheck(num_descriptors == 2, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Expected 2 descriptors, got %" PetscInt_FMT, num_descriptors);

  PetscCall(PetscStrcmp(names[0], "Help", &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong name for descriptor 0, expected 'Title' but got %s", names[0] ? names[0] : "(null)");

  PetscCall(PetscStrcmp(names[1], "Long Description", &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong name for descriptor 1, expected 'Long Description' but got %s", names[1] ? names[1] : "(null)");

  PetscCall(PetscStrcmp(descriptors[0], help, &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong value for descriptor 0, expected 'viewer tests ex8' but got %s", descriptors[0] ? descriptors[0] : "(null)");

  PetscCall(GetLongDescription(&expected_desc_len, &expected_desc));
  PetscCall(PetscStrcmp(descriptors[1], expected_desc, &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong value for descriptor 2");
  PetscCall(PetscFree(expected_desc));

  for (PetscInt i = 0; i < num_descriptors; i++) {
    PetscCall(PetscFree(descriptors[i]));
    PetscCall(PetscFree(names[i]));
  }
  PetscCall(PetscFree(descriptors));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscViewer viewer;
  DM          dm;
  Vec         v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));

  PetscCall(DMGetGlobalVector(dm, &v));
  PetscCall(VecZeroEntries(v));

  PetscCall(TestOpen(FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(v, viewer));
  PetscCall(TestWriteDescriptors(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ, &viewer));
  PetscCall(TestReadDescriptors(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(DMRestoreGlobalVector(dm, &v));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     nsize: 1 2
     output_file: output/empty.out
     args: -dm_plex_box_faces 3,3,3 -dm_plex_dim 3

TEST*/
