const char help[] = "Visualize PetscFE finite elements with SAWs";

#include <petsc.h>
#include <petscviewersaws.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmimpl.h>
#include <SAWs.h>

typedef struct _n_PetscFESAWsArrayLink *PetscFESAWsArrayLink;

struct _n_PetscFESAWsArrayLink
{
  void                *mem;
  PetscFESAWsArrayLink next;
};

static PetscErrorCode PetscFESAWsArrayLinkDestroy(PetscFESAWsArrayLink *link)
{
  PetscFunctionBegin;
  PetscCall(PetscFree((*link)->mem));
  PetscCall(PetscFree(*link));
  PetscFunctionReturn(0);
}

typedef struct _p_PetscFESAWs *PetscFESAWs;

struct _PetscFESAWsOps {};

struct _p_PetscFESAWs
{
  PETSCHEADER(struct _PetscFESAWsOps);
  size_t               dir_alloc;
  size_t               dir_tip;
  char                *dir;
  PetscFESAWsArrayLink arrays;
  int                  number_of_elements;
};

PetscClassId PETSCFESAWS_CLASSID;

static PetscErrorCode CreateEvaluationGrid(DMPolytopeType tope, PetscInt points_per_direction, PetscQuadrature *points)
{
  PetscFunctionBegin;
  switch (tope) {
  case DM_POLYTOPE_POINT:
    PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, points));
    PetscCall(PetscQuadratureSetData(*points, 0, 0, 0, NULL, NULL));
    break;
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    {
      PetscInt dim = DMPolytopeTypeGetDim(tope);
      PetscInt num_points;
      PetscInt k = points_per_direction - 1;
      PetscReal *x;
      PetscInt *bary;

      PetscCall(PetscDTBinomialInt(dim+k, k, &num_points));
      PetscCall(PetscCalloc1(num_points * dim, &x));
      PetscCall(PetscCalloc1(dim + 1, &bary));
      bary[dim] = k;
      for (PetscInt i = 0; i < num_points; i++) {
        // apply barycentric to biunit simplex transform
        for (PetscInt a = 0; a < dim+1; a++) {
          PetscReal val = (PetscReal) bary[a] / (PetscReal) k;
          for (PetscInt b = 0; b < dim; b++) {
            PetscReal mult = (a == b) ? 1. : -1.;
            x[i * dim + b] += val * mult;
          }
        }
        { // get the next barycentric coordinate
          PetscInt sum = k - bary[0] - bary[1];
          PetscInt d = 1;
          PetscInt tot;

          while (sum) {
            sum -= bary[++d];
          }
          tot = bary[d];
          bary[d-1]++;
          for (; d < dim; d++) {
            bary[d] = 0;
          }
          bary[dim] = tot-1;
        }
      }
      PetscCall(PetscFree(bary));
      PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, points));
      PetscCall(PetscQuadratureSetData(*points, dim, 0, num_points, x, NULL));
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_TRI_PRISM:
    {
      PetscQuadrature sub_quad, line_quad;
      DMPolytopeType sub_tope =
        (tope == DM_POLYTOPE_QUADRILATERAL) ? DM_POLYTOPE_SEGMENT :
        (tope == DM_POLYTOPE_HEXAHEDRON) ? DM_POLYTOPE_QUADRILATERAL :
        DM_POLYTOPE_TRIANGLE;
      PetscCall(CreateEvaluationGrid(sub_tope, points_per_direction, &sub_quad));
      PetscCall(CreateEvaluationGrid(DM_POLYTOPE_SEGMENT, points_per_direction, &line_quad));
      PetscCall(PetscDTTensorQuadratureCreate(sub_quad, line_quad, points));
      PetscCall(PetscQuadratureDestroy(&sub_quad));
      PetscCall(PetscQuadratureDestroy(&line_quad));
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscFESAWs does not create arrays of this type");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsCreateArray(PetscFESAWs fes, SAWs_Data_type dtype, size_t count, void *data_pointer)
{
  MPI_Datatype mpi_dtype = MPI_BYTE;
  PetscFESAWsArrayLink link;
  int          dsize;

  PetscFunctionBegin;
  switch (dtype) {
  case SAWs_STRING:
  case SAWs_CHAR:
    mpi_dtype = MPI_CHAR;
    break;
  case SAWs_INT:
    mpi_dtype = MPI_INT;
    break;
  case SAWs_FLOAT:
    mpi_dtype = MPI_FLOAT;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscFESAWs does not create arrays of this type");
  }
  PetscCallMPI(MPI_Type_size(mpi_dtype,&dsize));
  PetscCall(PetscNewLog(fes, &link));
  link->next = fes->arrays;
  fes->arrays = link;
  PetscCall(PetscMalloc(count*dsize, &(link->mem)));
  if (dtype == SAWs_STRING) {
    *((void **) data_pointer) = &(link->mem);
  } else {
    *((void **) data_pointer) = link->mem;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsWriteProperty(PetscFESAWs fes, const char *property, const void *data, int count, SAWs_Memory_type mtype, SAWs_Data_type dtype)
{
  size_t buf_size = fes->dir_alloc + BUFSIZ;
  size_t printed_size;
  char   *buf;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(buf_size, &buf));
  PetscCall(PetscSNPrintfCount(buf, buf_size, "%s/%s", &printed_size, fes->dir, property));
  if (printed_size >= buf_size) {
    PetscCall(PetscFree(buf));
    buf_size = printed_size + 1;
    PetscCall(PetscMalloc1(buf_size, &buf));
    PetscCall(PetscSNPrintfCount(buf, buf_size, "%s/%s", &printed_size, fes->dir, property));
  }
  PetscStackCallSAWs(SAWs_Register,(buf, (void *) data, count, mtype, dtype));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsDestroy(PetscFESAWs *fes)
{
  PetscBool completed = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscFESAWsWriteProperty(*fes, "completed", &completed, 1, SAWs_READ, SAWs_BOOLEAN));
  PetscCall(PetscObjectSAWsBlock((PetscObject) *fes));
  {
    PetscFESAWsArrayLink head = (*fes)->arrays;
    while (head) {
      PetscFESAWsArrayLink next = head->next;
      PetscCall(PetscFESAWsArrayLinkDestroy(&head));
      head = next;
    }
  }
  PetscCall(PetscObjectSAWsViewOff((PetscObject) *fes));
  PetscCall(PetscHeaderDestroy(fes));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsView_SAWs(PetscFESAWs fes, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectName((PetscObject)fes));
  PetscCall(PetscObjectViewSAWs((PetscObject) fes, viewer));
  PetscCall(PetscFESAWsWriteProperty(fes, "number_of_elements", &(fes->number_of_elements), 1, SAWs_READ, SAWs_INT ));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsCreate(MPI_Comm comm, PetscFESAWs *fes)
{
  PetscFunctionBegin;
  PetscCall(PetscHeaderCreate(*fes, PETSCFESAWS_CLASSID, "PetscFESAWs", "PetscFE SAWs Manager", "", comm, PetscFESAWsDestroy, PetscFESAWsView_SAWs));
  (*fes)->number_of_elements = 0;
  PetscCall(PetscObjectSAWsSetBlock((PetscObject) *fes, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsSetBaseDirectory(PetscFESAWs fes, const char dir[])
{
  PetscFunctionBegin;
  PetscCall(PetscFree(fes->dir));
  fes->dir_alloc = BUFSIZ;
  PetscCall(PetscMalloc1(fes->dir_alloc, &(fes->dir)));
  PetscCall(PetscSNPrintfCount(fes->dir, fes->dir_alloc, "/PETSc/%s/", &(fes->dir_tip), dir));
  fes->dir_tip--;
  if (fes->dir_tip >= fes->dir_alloc) {
    PetscCall(PetscFree(fes->dir));
    fes->dir_alloc = (fes->dir_tip + 1) * 2;
    PetscCall(PetscMalloc1(fes->dir_alloc, &(fes->dir)));
    PetscCall(PetscSNPrintfCount(fes->dir, fes->dir_alloc, "/PETSc/%s/", &(fes->dir_tip), dir));
    fes->dir_tip--;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsDirectoryPush(PetscFESAWs fes, const char dir[])
{
  size_t tip_inc;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintfCount(&fes->dir[fes->dir_tip], fes->dir_alloc - fes->dir_tip, "%s/", &tip_inc, dir));
  tip_inc--;
  if (fes->dir_tip + tip_inc >= fes->dir_alloc) {
    const char *dir_old = fes->dir;

    fes->dir_alloc = (fes->dir_tip + tip_inc + 1) * 2;
    PetscCall(PetscMalloc1(fes->dir_alloc, &(fes->dir)));
    PetscCall(PetscArraycpy(fes->dir, dir_old, fes->dir_tip));
    PetscCall(PetscSNPrintfCount(&fes->dir[fes->dir_tip], fes->dir_alloc - fes->dir_tip, "%s/", &tip_inc, dir));
    tip_inc--;
    PetscCall(PetscFree(dir_old));
  }
  fes->dir_tip += tip_inc;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsDirectoryPop(PetscFESAWs fes)
{
  PetscFunctionBegin;
  while (fes->dir_tip > 0 && fes->dir[fes->dir_tip-1] == '/') fes->dir_tip--;
  while (fes->dir_tip > 0 && fes->dir[fes->dir_tip-1] != '/') fes->dir_tip--;
  fes->dir[fes->dir_tip] = '\0';
  PetscFunctionReturn(0);
}

PetscFESAWs PetscFESAWsManager = NULL;

static PetscErrorCode PetscFESAWsViewReferenceElement(PetscFESAWs fes, DM refel)
{
  PetscInt num_points;
  PetscInt v_start, v_end;
  PetscInt dim;
  int num_coords, offset;
  int *i_array;
  Vec coord_vec;
  float *coords;
  PetscScalar *these_coords;
  DM coord_dm;

  PetscFunctionBegin;
  PetscCall(DMPlexGetChart(refel, NULL, &num_points));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 2*num_points + 1, &i_array));
  i_array[2*num_points] = num_points;
  PetscCall(PetscFESAWsWriteProperty(fes, "number_of_mesh_points", &i_array[2*num_points], 1, SAWs_READ, SAWs_INT));

  PetscCall(DMGetCoordinateDim(refel, &dim));
  PetscCall(DMPlexGetDepthStratum(refel, 0, &v_start, &v_end));
  num_coords = 0;
  for (PetscInt v = v_start; v < v_end; v++) {
    PetscInt star_size = 0;
    PetscInt *star = NULL;

    PetscCall(DMPlexGetTransitiveClosure(refel, v, PETSC_FALSE, &star_size, &star));
    num_coords += star_size * dim;
    PetscCall(DMPlexRestoreTransitiveClosure(refel, v, PETSC_FALSE, &star_size, &star));
  }
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_FLOAT, num_coords * dim, &coords));
  PetscCall(DMGetCoordinatesLocalNoncollective(refel, &coord_vec));
  PetscCall(DMGetCoordinateDM(refel, &coord_dm));
  PetscCall(PetscMalloc1(dim == 0 ? 1 : num_coords * dim, &these_coords));
  offset = 0;
  for (PetscInt p = 0; p < num_points; p++) {
    DMPolytopeType ptype;
    char point_string[3];
    PetscInt these_coords_size = dim == 0 ? 1 : num_coords * dim;
    PetscInt csize = these_coords_size;
    PetscInt p_num_verts;
    PetscInt p_dim;
    PetscInt coord_start = offset;

    PetscCall(PetscSNPrintf(point_string, 3, "%d", p));
    PetscCall(PetscFESAWsDirectoryPush(fes, point_string));
    PetscCall(DMPlexGetCellType(refel, p, &ptype));
    PetscCall(DMPlexGetPointDepth(refel, p, &p_dim));
    PetscCall(PetscFESAWsWriteProperty(fes, "polytope", &DMPolytopeTypes[ptype], 1, SAWs_READ, SAWs_STRING));
    PetscCall(DMPlexVecGetClosure(coord_dm, NULL, coord_vec, p, &csize, &these_coords));
    i_array[2*p] = p_num_verts = (dim == 0 ? 1 : (csize / dim));
    i_array[2*p+1] = p_dim;
    for (PetscInt d = 0; d < dim; d++) {
      for (PetscInt v = 0; v < p_num_verts; v++) {
        coords[offset++] = PetscRealPart(these_coords[v * dim + d]);
      }
    }
    PetscCall(PetscFESAWsWriteProperty(fes, "number_of_vertices", &i_array[2*p], 1, SAWs_READ, SAWs_INT));
    PetscCall(PetscFESAWsWriteProperty(fes, "dimension", &i_array[2*p+1], 1, SAWs_READ, SAWs_INT));
    PetscCall(PetscFESAWsWriteProperty(fes, "coordinates", &coords[coord_start], csize, SAWs_READ, SAWs_FLOAT));
    PetscCall(PetscFESAWsDirectoryPop(fes));
  }
  PetscCall(PetscFree(these_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsViewBasisSpace(PetscFESAWs fes, PetscSpace sp, DMPolytopeType tope)
{
  PetscInt min_degree, max_degree;
  PetscQuadrature evaluation_points;
  PetscInt dim, num_points;
  PetscInt *i_array;
  const PetscReal *points;
  float *i_points;

  PetscFunctionBegin;
  PetscCall(PetscSpaceGetDegree(sp, &min_degree, &max_degree));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 3, &i_array));
  i_array[0] = min_degree;
  i_array[1] = max_degree;
  PetscCall(PetscFESAWsWriteProperty(fes, "minimum_degree", &i_array[0], 1, SAWs_READ, SAWs_INT));
  PetscCall(PetscFESAWsWriteProperty(fes, "maximum_degree", &i_array[1], 1, SAWs_READ, SAWs_INT));
  PetscCall(CreateEvaluationGrid(tope, PetscMax(2,1 + 2*max_degree), &evaluation_points));
  PetscCall(PetscQuadratureView(evaluation_points, NULL));
  PetscCall(PetscQuadratureGetData(evaluation_points, &dim, NULL, &num_points, &points, NULL));
  i_array[2] = num_points;
  PetscCall(PetscFESAWsWriteProperty(fes, "number_of_grid_points", &i_array[2], 1, SAWs_READ, SAWs_INT));
  printf("num_points %d dim %d\n", num_points, dim);
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_FLOAT, num_points * dim, &i_points));
  for (PetscInt p = 0; p < num_points; p++) {
    for (PetscInt d = 0; d < dim; d++) {
      i_points[d * num_points + p] = points[p * dim + d];
      printf("%+7.2g ", points[p * dim + d]);
    }
    printf("\n");
  }
  PetscCall(PetscFESAWsWriteProperty(fes, "grid_points", &i_points, num_points *dim, SAWs_READ, SAWs_FLOAT));
  PetscCall(PetscQuadratureDestroy(&evaluation_points));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetFunctionalPoint(PetscDualSpace dsp, PetscInt f, PetscInt *point)
{
  PetscInt     p_start, p_end;
  PetscSection sec;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetSection(dsp, &sec));
  PetscCall(PetscSectionGetChart(sec, &p_start, &p_end));
  for (PetscInt p = p_start; p < p_end; p++) {
    PetscInt dof, off;

    PetscCall(PetscSectionGetDof(sec, p, &dof));
    PetscCall(PetscSectionGetOffset(sec, p, &off));
    if (off <= f && f < off + dof) {
      *point = p;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Functional out of range");
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsViewDualSpace(PetscFESAWs fes, PetscDualSpace dsp)
{
  DM          dm;
  PetscInt    form_degree, dim;
  PetscBool   continuous;
  size_t      var_len;
  char      **variance;
  const char *variance_static;
  const char *variances[] = {
    "invariant | \\\\(H^1\\\\) | \\\\(0\\\\)-form",
    "covariant | \\\\(H(\\\\mathrm{curl})\\\\) | \\\\(1\\\\)-form", 
    "contravariant | \\\\(H(\\\\mathrm{div})\\\\) | \\\\((\\\\star X)\\\\)-form", 
    "\\\\(L^2\\\\) | \\\\(X\\\\)-form",
    "\\\\(X\\\\)-form"
  };
  int          continuity_index;
  static const char *continuities[] = {
    "discontinuous", 
    "continuous", 
    "tangentially continuous", 
    "normally continuous", 
    "trace continuous", 
  };
  static const char *types[] = {
    "nodal",
    "modal"
  };
  const char  dims[] = "0123456789";
  float *points_and_weights;
  int *sizes_and_points;
  PetscInt Nb, Nc;
  PetscInt Np;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(dsp, &dm));
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(PetscDualSpaceGetFormDegree(dsp, &form_degree));
  PetscCall(PetscDualSpaceLagrangeGetContinuity(dsp, &continuous));
  variance_static =
    (form_degree == 0) ? variances[0] :
    (form_degree == dim) ? variances[3] :
    (form_degree == 1) ? variances[1] :
    (form_degree == -(dim-1)) ? variances[2] :
    variances[4];
  PetscCall(PetscStrlen(variance_static, &var_len));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_STRING, var_len + 1, &variance));
  PetscCall(PetscArraycpy(*variance, variance_static, var_len));
  (*variance)[var_len] = '\0';
  for (size_t i = 0; i < var_len; i++) {
    if ((*variance)[i] == 'X') (*variance)[i] = dims[PetscAbsInt(form_degree)];
  }
  PetscCall(PetscFESAWsWriteProperty(fes, "variance", variance, 1, SAWs_READ, SAWs_STRING));
  // TODO: detect modal
  PetscCall(PetscFESAWsWriteProperty(fes, "type", &types[0], 1, SAWs_READ, SAWs_STRING));

  continuity_index =
    (!continuous) ? 0 :
    (form_degree == 0) ? 1 :
    (form_degree == 1) ? 2 :
    (form_degree == -(dim-1)) ? 3 :
    4;
  PetscCall(PetscFESAWsWriteProperty(fes, "continuity", &continuities[continuity_index], 1, SAWs_READ, SAWs_STRING));

  PetscCall(PetscDualSpaceGetDimension(dsp, &Nb));
  PetscCall(PetscDualSpaceGetNumComponents(dsp, &Nc));
  Np = 0;
  for (PetscInt i = 0; i < Nb; i++) {
    PetscQuadrature f;
    PetscInt        fNp;

    PetscCall(PetscDualSpaceGetFunctional(dsp, i, &f));
    PetscCall(PetscQuadratureGetData(f, NULL, NULL, &fNp, NULL, NULL));
    Np += fNp;
  }
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_FLOAT, (Nc + dim) * Np, &points_and_weights));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 2* Nb + 1, &sizes_and_points));
  sizes_and_points[2*Nb] = form_degree;
  PetscCall(PetscFESAWsWriteProperty(fes, "form_degree", &sizes_and_points[2*Nb], 1, SAWs_READ, SAWs_INT));
  for (PetscInt i = 0; i < Nb; i++) {
    PetscQuadrature f;
    char functional_string[5];
    const PetscReal *f_points;
    const PetscReal *f_weights;
    PetscInt fNp, mesh_point;
    float *points, *weights;

    PetscCall(PetscSNPrintf(functional_string, 5, "%d", i));
    PetscCall(PetscFESAWsDirectoryPush(fes, functional_string));
    PetscCall(PetscDualSpaceGetFunctional(dsp, i, &f));
    PetscCall(PetscQuadratureGetData(f, NULL, NULL, &fNp, &f_points, &f_weights));
    PetscCall(PetscDualSpaceGetFunctionalPoint(dsp, i, &mesh_point));
    sizes_and_points[2*i] = fNp;
    sizes_and_points[2*i + 1] = mesh_point;
    PetscCall(PetscFESAWsWriteProperty(fes, "number_of_nodes", &sizes_and_points[2*i], 1, SAWs_READ, SAWs_INT));
    PetscCall(PetscFESAWsWriteProperty(fes, "mesh_point", &sizes_and_points[2*i+1], 1, SAWs_READ, SAWs_INT));

    points = &points_and_weights[0];
    points_and_weights += fNp * dim;
    for (PetscInt d = 0; d < dim; d++) {
      for (PetscInt p = 0; p < fNp; p++) {
        points[d * fNp + p] = f_points[p * dim + d];
      }
    }
    PetscCall(PetscFESAWsWriteProperty(fes, "nodes", points, dim * fNp, SAWs_READ, SAWs_FLOAT));

    weights = &points_and_weights[0];
    points_and_weights += fNp * Nc;
    for (PetscInt w = 0; w < Nc; w++) {
      for (PetscInt p = 0; p < fNp; p++) {
        weights[w * fNp + p] = f_weights[p * dim + w];
      }
    }
    PetscCall(PetscFESAWsWriteProperty(fes, "weights", weights, Nc * fNp, SAWs_READ, SAWs_FLOAT));
    PetscCall(PetscFESAWsDirectoryPop(fes));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsAddFE(PetscFESAWs fes, PetscFE fe)
{
  char num_string[3];

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(num_string, 3, "%d", fes->number_of_elements));
  PetscCall(PetscFESAWsDirectoryPush(fes, num_string));
  fes->number_of_elements++;
  {
    PetscInt dim, Nb, Nc;
    PetscSpace space;
    PetscDualSpace dsp;
    DMPolytopeType tope;
    DM refel;
    int *i_dim_Nb_Nc;
    const char *name;
    const char *prefix;
    char **aname, **aprefix;
    size_t name_len, prefix_len;

    PetscCall(PetscObjectName((PetscObject) fe));
    PetscCall(PetscObjectGetName((PetscObject) fe, &name));
    PetscCall(PetscStrlen(name, &name_len));
    PetscCall(PetscFESAWsCreateArray(fes, SAWs_STRING, name_len+1, &aname));
    PetscCall(PetscArraycpy(*aname, name, name_len+1));
    (*aname)[name_len] = '\0';
    PetscCall(PetscFESAWsWriteProperty(fes, "name", aname, 1, SAWs_READ, SAWs_STRING));

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject) fe, &prefix));
    PetscCall(PetscStrlen(prefix, &prefix_len));
    PetscCall(PetscFESAWsCreateArray(fes, SAWs_STRING, prefix_len+1, &aprefix));
    PetscCall(PetscArraycpy(*aprefix, prefix, prefix_len));
    (*aprefix)[prefix_len] = '\0';
    PetscCall(PetscFESAWsWriteProperty(fes, "options_prefix", aprefix, 1, SAWs_READ, SAWs_STRING));

    PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 3, &i_dim_Nb_Nc));

    PetscCall(PetscFEGetSpatialDimension(fe, &dim));
    i_dim_Nb_Nc[0] = dim;
    PetscCall(PetscFESAWsWriteProperty(fes, "spatial_dimension", &i_dim_Nb_Nc[0], 1, SAWs_READ, SAWs_INT));

    PetscCall(PetscFEGetDimension(fe, &Nb));
    i_dim_Nb_Nc[1] = Nb;
    PetscCall(PetscFESAWsWriteProperty(fes, "dimension", &i_dim_Nb_Nc[1], 1, SAWs_READ, SAWs_INT));

    PetscCall(PetscFEGetNumComponents(fe, &Nc));
    i_dim_Nb_Nc[2] = Nc;
    PetscCall(PetscFESAWsWriteProperty(fes, "number_of_components", &i_dim_Nb_Nc[2], 1, SAWs_READ, SAWs_INT));

    PetscCall(PetscFEGetDualSpace(fe, &dsp));
    PetscCall(PetscFESAWsDirectoryPush(fes, "dual_space"));
    PetscCall(PetscFESAWsViewDualSpace(fes, dsp));
    PetscCall(PetscFESAWsDirectoryPop(fes));

    PetscCall(PetscDualSpaceGetDM(dsp, &refel));
    PetscCall(DMPlexGetCellType(refel, 0, &tope));
    PetscCall(PetscFESAWsDirectoryPush(fes, "reference_element"));
    PetscCall(PetscFESAWsViewReferenceElement(fes, refel));
    PetscCall(PetscFESAWsDirectoryPop(fes));

    PetscCall(PetscFEGetBasisSpace(fe, &space));
    PetscCall(PetscFESAWsDirectoryPush(fes, "basis_space"));
    PetscCall(PetscFESAWsViewBasisSpace(fes, space, tope));
    PetscCall(PetscFESAWsDirectoryPop(fes));
  }
  PetscCall(PetscFESAWsDirectoryPop(fes));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEView_SAWs(PetscFE fe, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (!PetscFESAWsManager) {
    PetscCall(PetscFESAWsCreate(PETSC_COMM_WORLD, &PetscFESAWsManager));
    PetscCall(PetscObjectSetName((PetscObject) PetscFESAWsManager, "PetscFESAWs"));
    PetscCall(PetscFESAWsSetBaseDirectory(PetscFESAWsManager, "FE"));
    PetscCall(PetscFESAWsView_SAWs(PetscFESAWsManager, viewer));
    PetscCall(PetscObjectRegisterDestroy((PetscObject) PetscFESAWsManager));
  }
  PetscFESAWsAddFE(PetscFESAWsManager, fe);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSpace(DMPolytopeType tope, const char prefix[], PetscInt Nc, PetscSpace *sp)
{
  PetscSpace scalar_sp;
  PetscInt dim = DMPolytopeTypeGetDim(tope);

  PetscFunctionBegin;
  PetscCall(PetscSpaceCreate(PETSC_COMM_WORLD, &scalar_sp));
  PetscCall(PetscSpaceSetNumVariables(scalar_sp, dim));
  PetscCall(PetscSpaceSetNumComponents(scalar_sp, 1));
  switch (tope) {
  case DM_POLYTOPE_POINT:
    PetscCall(PetscSpaceSetDegree(scalar_sp, 0, 0));
    break;
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(PetscSpaceSetDegree(scalar_sp, 1 + dim, 1 + dim));
    break;
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
    {
      PetscSpace space_1d;
      PetscCall(PetscSpaceSetType(scalar_sp, PETSCSPACETENSOR));
      PetscCall(PetscSpaceTensorSetNumSubspaces(scalar_sp, dim));
      PetscCall(PetscSpaceCreate(PETSC_COMM_WORLD, &space_1d));
      PetscCall(PetscSpaceSetNumVariables(space_1d, 1));
      PetscCall(PetscSpaceSetNumComponents(space_1d, 1));
      PetscCall(PetscSpaceSetDegree(space_1d, 2, 2));
      for (PetscInt d = 0; d < dim; d++) {
        PetscCall(PetscSpaceTensorSetSubspace(scalar_sp, d, space_1d));
      }
      PetscCall(PetscSpaceDestroy(&space_1d));
    }
    break;
  case DM_POLYTOPE_TRI_PRISM:
    {
      PetscSpace space_1d, space_2d;
      PetscCall(PetscSpaceSetType(scalar_sp, PETSCSPACETENSOR));
      PetscCall(PetscSpaceTensorSetNumSubspaces(scalar_sp, 2));

      PetscCall(PetscSpaceCreate(PETSC_COMM_WORLD, &space_1d));
      PetscCall(PetscSpaceSetNumVariables(space_1d, 1));
      PetscCall(PetscSpaceSetNumComponents(space_1d, 1));
      PetscCall(PetscSpaceSetDegree(space_1d, 2, 2));

      PetscCall(PetscSpaceCreate(PETSC_COMM_WORLD, &space_2d));
      PetscCall(PetscSpaceSetNumVariables(space_2d, 2));
      PetscCall(PetscSpaceSetNumComponents(space_2d, 1));
      PetscCall(PetscSpaceSetDegree(space_2d, 2, 2));

      PetscCall(PetscSpaceTensorSetSubspace(scalar_sp, 0, space_1d));
      PetscCall(PetscSpaceTensorSetSubspace(scalar_sp, 1, space_2d));

      PetscCall(PetscSpaceDestroy(&space_1d));
      PetscCall(PetscSpaceDestroy(&space_2d));
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "unhandled polytope %s", DMPolytopeTypes[tope]);
  }
  if (Nc == 1) {
    *sp = scalar_sp;
  } else {
    PetscCall(PetscSpaceCreate(PETSC_COMM_WORLD, sp));
    PetscCall(PetscSpaceSetType(*sp, PETSCSPACESUM));
    PetscCall(PetscSpaceSetNumVariables(*sp, dim));
    PetscCall(PetscSpaceSetNumComponents(*sp, Nc));
    PetscCall(PetscSpaceSumSetNumSubspaces(*sp, dim));
    PetscCall(PetscSpaceSumSetConcatenate(*sp, PETSC_TRUE));
    for (PetscInt d = 0; d < dim; d++) {
      PetscCall(PetscSpaceSumSetSubspace(*sp, d, scalar_sp));
    }
    PetscCall(PetscSpaceDestroy(&scalar_sp));
  }
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*sp, prefix));
  PetscCall(PetscSpaceSetFromOptions(*sp));
  PetscCall(PetscSpaceSetUp(*sp));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDualSpace(DM refel, DMPolytopeType tope, const char prefix[], PetscInt Nc, PetscInt form_degree, PetscDualSpace *dsp)
{
  PetscInt dim = DMPolytopeTypeGetDim(tope);

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceCreate(PETSC_COMM_WORLD, dsp));
  PetscCall(PetscDualSpaceSetDM(*dsp, refel));
  PetscCall(PetscDualSpaceSetType(*dsp, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceSetFormDegree(*dsp, form_degree));
  PetscCall(PetscDualSpaceSetNumComponents(*dsp, Nc));
  switch (tope) {
  case DM_POLYTOPE_POINT:
    PetscCall(PetscDualSpaceSetOrder(*dsp, 0));
    break;
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(PetscDualSpaceSetOrder(*dsp, 1 + dim));
    PetscCall(PetscDualSpaceLagrangeSetTensor(*dsp, PETSC_FALSE));
    break;
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_TRI_PRISM:
    PetscCall(PetscDualSpaceSetOrder(*dsp, 2));
    PetscCall(PetscDualSpaceLagrangeSetTensor(*dsp, PETSC_TRUE));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "unhandled polytope %s", DMPolytopeTypes[tope]);
  }
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dsp, prefix));
  PetscCall(PetscDualSpaceSetFromOptions(*dsp));
  PetscCall(PetscDualSpaceSetUp(*dsp));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscClassIdRegister("PetscFE SAWs Manager",&PETSCFESAWS_CLASSID));
  DMPolytopeType topes[] = {
    DM_POLYTOPE_POINT,
    DM_POLYTOPE_SEGMENT,
    DM_POLYTOPE_TRIANGLE,
    DM_POLYTOPE_QUADRILATERAL,
    DM_POLYTOPE_TETRAHEDRON,
    DM_POLYTOPE_HEXAHEDRON,
    DM_POLYTOPE_TRI_PRISM
  };
  size_t num_topes = sizeof(topes) / sizeof(topes[0]);
  for (size_t t = 0; t < num_topes; t++) {
    DM refel;
    PetscInt dim;
    PetscInt lo;

    PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_WORLD, topes[t], &refel));
    PetscCall(DMGetCoordinateDim(refel, &dim));
    lo = (dim > 1) ? -1 : 0;
    for (PetscInt form_degree = lo; form_degree <= dim; form_degree++) {
      PetscInt fd = form_degree == -1 ? -(dim - 1) : form_degree;
      PetscInt Nc;
      PetscSpace sp;
      PetscDualSpace dsp;
      PetscFE fe;
      char prefix[BUFSIZ];
      char name[BUFSIZ];

      PetscCall(PetscSNPrintf(prefix, BUFSIZ, "%s_%" PetscInt_FMT "_", DMPolytopeTypes[topes[t]], fd));
      PetscCall(PetscSNPrintf(name, BUFSIZ, "%s(%" PetscInt_FMT ")", DMPolytopeTypes[topes[t]], fd));
      PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(fd), &Nc));
      PetscCall(CreateSpace(topes[t], prefix, Nc, &sp));
      PetscCall(CreateDualSpace(refel, topes[t], prefix, Nc, fd, &dsp));
      PetscCall(PetscFECreate(PETSC_COMM_WORLD, &fe));
      PetscCall(PetscObjectSetName((PetscObject) fe, name));
      PetscCall(PetscFESetNumComponents(fe, Nc));
      PetscCall(PetscFESetBasisSpace(fe, sp));
      PetscCall(PetscFESetDualSpace(fe, dsp));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fe, prefix));
      PetscCall(PetscFESetFromOptions(fe));
      PetscCall(PetscFESetUp(fe));
      PetscCall(PetscFEView_SAWs(fe, PETSC_VIEWER_SAWS_(PETSC_COMM_WORLD)));
      PetscCall(PetscFEDestroy(&fe));
      PetscCall(PetscDualSpaceDestroy(&dsp));
      PetscCall(PetscSpaceDestroy(&sp));
    }
    PetscCall(DMDestroy(&refel));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -saws_root $PETSC_DIR/share/petsc/saws/

TEST*/
