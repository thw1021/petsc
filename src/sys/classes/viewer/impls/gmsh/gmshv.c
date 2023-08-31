
#include <petsc/private/viewergmshimpl.h> /*I "petscviewer.h" I*/

static PetscErrorCode PetscViewerView_GMSH(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_GMSH *gmsh = (PetscViewer_GMSH *)v->data;

  PetscFunctionBegin;
  if (gmsh->filename) PetscCall(PetscViewerASCIIPrintf(viewer, "Filename: %s\n", gmsh->filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_GMSH(PetscViewer viewer)
{
  PetscViewer_GMSH *gmsh = (PetscViewer_GMSH *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&gmsh->viewer));
  PetscCall(PetscFree(gmsh->filename));
  PetscCall(PetscFree(gmsh));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetMode_GMSH(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_GMSH *gmsh = (PetscViewer_GMSH *)viewer->data;

  PetscFunctionBegin;
  gmsh->btype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetMode_GMSH(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_GMSH *gmsh = (PetscViewer_GMSH *)viewer->data;

  PetscFunctionBegin;
  *type = gmsh->btype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* these will eventually not be static and will replace the ones in plexgmsh.c */
static PetscErrorCode GmshReadString(PetscViewer_GMSH *gmsh, char *buf, PetscInt count)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerRead(gmsh->viewer, buf, count, NULL, PETSC_STRING));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GmshMatch(PETSC_UNUSED PetscViewer_GMSH *gmsh, const char Section[], char line[PETSC_MAX_PATH_LEN], PetscBool *match)
{
  PetscFunctionBegin;
  PetscCall(PetscStrcmp(line, Section, match));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GmshExpect(PetscViewer_GMSH *gmsh, const char Section[], char line[PETSC_MAX_PATH_LEN])
{
  PetscBool match;

  PetscFunctionBegin;
  PetscCall(GmshMatch(gmsh, Section, line, &match));
  PetscCheck(match, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file, expecting %s", Section);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GmshReadSection(PetscViewer_GMSH *gmsh, char line[PETSC_MAX_PATH_LEN])
{
  PetscBool match;

  PetscFunctionBegin;
  while (PETSC_TRUE) {
    PetscCall(GmshReadString(gmsh, line, 1));
    PetscCall(GmshMatch(gmsh, "$Comments", line, &match));
    if (!match) break;
    while (PETSC_TRUE) {
      PetscCall(GmshReadString(gmsh, line, 1));
      PetscCall(GmshMatch(gmsh, "$EndComments", line, &match));
      if (match) break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetName_GMSH(PetscViewer viewer, const char *filename)
{
  PetscViewer_GMSH *gmsh = (PetscViewer_GMSH *)viewer->data;
  PetscMPIInt       rank;
  int               fileType;
  PetscViewerType   vtype;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscCall(PetscFree(gmsh->filename));
  PetscCall(PetscStrallocpy(filename, &gmsh->filename));
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* Determine Gmsh file type (ASCII or binary) from file header */
  if (rank == 0) {
    char  line[PETSC_MAX_PATH_LEN];
    int   snum;
    float version;
    int   fileFormat;

    PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &gmsh->viewer));
    PetscCall(PetscViewerSetType(gmsh->viewer, PETSCVIEWERASCII));
    PetscCall(PetscViewerFileSetMode(gmsh->viewer, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(gmsh->viewer, filename));
    /* Read only the first two lines of the Gmsh file */
    PetscCall(GmshReadSection(gmsh, line));
    PetscCall(GmshExpect(gmsh, "$MeshFormat", line));
    PetscCall(GmshReadString(gmsh, line, 2));
    snum       = sscanf(line, "%f %d", &version, &fileType);
    fileFormat = (int)roundf(version * 10);
    PetscCheck(snum == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to parse Gmsh file header: %s", line);
    PetscCheck(fileFormat >= 22, PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at least 2.2", (double)version);
    PetscCheck((int)version != 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f not supported", (double)version);
    PetscCheck(fileFormat <= 41, PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at most 4.1", (double)version);
    PetscCall(PetscViewerDestroy(&gmsh->viewer));
  }
  PetscCallMPI(MPI_Bcast(&fileType, 1, MPI_INT, 0, comm));
  vtype = (fileType == 0) ? PETSCVIEWERASCII : PETSCVIEWERBINARY;

  /* Create appropriate viewer and build plex */
  PetscCall(PetscViewerCreate(comm, &gmsh->viewer));
  PetscCall(PetscViewerSetType(gmsh->viewer, vtype));
  PetscCall(PetscViewerFileSetMode(gmsh->viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(gmsh->viewer, filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetName_GMSH(PetscViewer viewer, const char **filename)
{
  PetscViewer_GMSH *gmsh = (PetscViewer_GMSH *)viewer->data;

  PetscFunctionBegin;
  *filename = gmsh->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCVIEWERCGMSH - A viewer for GMSH files

  Level: beginner

  Note:
  This currently only works with `DMLoad()`, not with `DMView()`

  Developer Note:
  GMSH prescribes a file format. PETSc does not use an external package to access GMSH files but rather access the file formats directly. The file format may be ASCII or
  binary, hence this viewer merely contains a filename and an internal `PetscViewer` that provides file access.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `VecView()`, `DMView()`, `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `TSSetFromOptions()`
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_GMSH(PetscViewer v)
{
  PetscViewer_GMSH *gmsh;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gmsh));

  v->data         = gmsh;
  v->ops->destroy = PetscViewerDestroy_GMSH;
  v->ops->view    = PetscViewerView_GMSH;
  gmsh->btype     = FILE_MODE_UNDEFINED;
  gmsh->filename  = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_GMSH));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_GMSH));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_GMSH));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_GMSH));
  PetscFunctionReturn(PETSC_SUCCESS);
}
