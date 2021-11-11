static char help[] = "The main goal of this code is to retrieve the original element numbers as found in the "
                     "initial partitions (sInitialPartition)... but after the call to DMPlexDistribute";

#include <petsc.h>

const PetscReal sCoords7x10Mesh[88][2] = {
{0.00000000000000000e+00,0.00000000000000000e+00},
{2.00000000000000000e+00,0.00000000000000000e+00},
{0.00000000000000000e+00,1.00000000000000000e+00},
{2.00000000000000000e+00,1.00000000000000000e+00},
{1.99999999999629308e-01,0.00000000000000000e+00},
{3.99999999999115674e-01,0.00000000000000000e+00},
{5.99999999998532818e-01,0.00000000000000000e+00},
{7.99999999997949796e-01,0.00000000000000000e+00},
{9.99999999997388533e-01,0.00000000000000000e+00},
{1.19999999999789408e+00,0.00000000000000000e+00},
{1.39999999999842006e+00,0.00000000000000000e+00},
{1.59999999999894693e+00,0.00000000000000000e+00},
{1.79999999999947402e+00,0.00000000000000000e+00},
{1.99999999999629308e-01,1.00000000000000000e+00},
{3.99999999999115674e-01,1.00000000000000000e+00},
{5.99999999998532818e-01,1.00000000000000000e+00},
{7.99999999997949796e-01,1.00000000000000000e+00},
{9.99999999997388533e-01,1.00000000000000000e+00},
{1.19999999999789408e+00,1.00000000000000000e+00},
{1.39999999999842006e+00,1.00000000000000000e+00},
{1.59999999999894693e+00,1.00000000000000000e+00},
{1.79999999999947402e+00,1.00000000000000000e+00},
{0.00000000000000000e+00,1.42857142857142794e-01},
{0.00000000000000000e+00,2.85714285714285698e-01},
{0.00000000000000000e+00,4.28571428571428492e-01},
{0.00000000000000000e+00,5.71428571428571397e-01},
{0.00000000000000000e+00,7.14285714285714302e-01},
{0.00000000000000000e+00,8.57142857142857095e-01},
{2.00000000000000000e+00,1.42857142857142794e-01},
{2.00000000000000000e+00,2.85714285714285698e-01},
{2.00000000000000000e+00,4.28571428571428492e-01},
{2.00000000000000000e+00,5.71428571428571397e-01},
{2.00000000000000000e+00,7.14285714285714302e-01},
{2.00000000000000000e+00,8.57142857142857095e-01},
{1.99999999999629391e-01,1.42857142857142794e-01},
{1.99999999999629391e-01,2.85714285714285698e-01},
{1.99999999999629391e-01,4.28571428571428714e-01},
{1.99999999999629391e-01,5.71428571428571397e-01},
{1.99999999999629391e-01,7.14285714285714302e-01},
{1.99999999999629391e-01,8.57142857142857095e-01},
{3.99999999999115674e-01,1.42857142857142794e-01},
{3.99999999999115674e-01,2.85714285714285698e-01},
{3.99999999999115674e-01,4.28571428571428714e-01},
{3.99999999999115674e-01,5.71428571428571397e-01},
{3.99999999999115674e-01,7.14285714285714302e-01},
{3.99999999999115674e-01,8.57142857142857095e-01},
{5.99999999998532596e-01,1.42857142857142794e-01},
{5.99999999998532818e-01,2.85714285714285698e-01},
{5.99999999998532818e-01,4.28571428571428603e-01},
{5.99999999998532818e-01,5.71428571428571397e-01},
{5.99999999998532818e-01,7.14285714285714302e-01},
{5.99999999998532818e-01,8.57142857142857206e-01},
{7.99999999997949685e-01,1.42857142857142794e-01},
{7.99999999997950018e-01,2.85714285714285698e-01},
{7.99999999997949685e-01,4.28571428571428714e-01},
{7.99999999997949796e-01,5.71428571428571397e-01},
{7.99999999997950018e-01,7.14285714285714191e-01},
{7.99999999997950018e-01,8.57142857142856984e-01},
{9.99999999997388533e-01,1.42857142857142794e-01},
{9.99999999997388533e-01,2.85714285714285698e-01},
{9.99999999997388533e-01,4.28571428571428714e-01},
{9.99999999997388533e-01,5.71428571428571397e-01},
{9.99999999997388533e-01,7.14285714285714302e-01},
{9.99999999997388533e-01,8.57142857142857095e-01},
{1.19999999999789408e+00,1.42857142857142794e-01},
{1.19999999999789408e+00,2.85714285714285698e-01},
{1.19999999999789408e+00,4.28571428571428492e-01},
{1.19999999999789408e+00,5.71428571428571397e-01},
{1.19999999999789408e+00,7.14285714285714302e-01},
{1.19999999999789408e+00,8.57142857142857095e-01},
{1.39999999999842006e+00,1.42857142857142794e-01},
{1.39999999999842006e+00,2.85714285714285698e-01},
{1.39999999999842006e+00,4.28571428571428714e-01},
{1.39999999999842006e+00,5.71428571428571397e-01},
{1.39999999999842006e+00,7.14285714285714302e-01},
{1.39999999999842006e+00,8.57142857142857095e-01},
{1.59999999999894693e+00,1.42857142857142794e-01},
{1.59999999999894693e+00,2.85714285714285698e-01},
{1.59999999999894693e+00,4.28571428571428714e-01},
{1.59999999999894693e+00,5.71428571428571397e-01},
{1.59999999999894693e+00,7.14285714285714302e-01},
{1.59999999999894693e+00,8.57142857142857095e-01},
{1.79999999999947402e+00,1.42857142857142794e-01},
{1.79999999999947291e+00,2.85714285714285698e-01},
{1.79999999999947402e+00,4.28571428571428492e-01},
{1.79999999999947402e+00,5.71428571428571397e-01},
{1.79999999999947402e+00,7.14285714285714302e-01},
{1.79999999999947402e+00,8.57142857142857095e-01}
};

/* For a 7 x 10 mesh, we have

   - 70 quads
   - 157 edges
   - 88 vertices

  Dividing in y, we have

   - 35 / 35 quads
   - 82 / 82 edges
   - 48 / 48 vertices
*/
//Connectivity of a 7x10 rectangular mesh of quads :
const PetscInt sConnectivity7x10Mesh[70][4] = {
{0,4,34,22},
{22,34,35,23},
{23,35,36,24},
{24,36,37,25},
{25,37,38,26},
{26,38,39,27},
{27,39,13,2},
{4,5,40,34},
{34,40,41,35},
{35,41,42,36},
{36,42,43,37},
{37,43,44,38},
{38,44,45,39},
{39,45,14,13},
{5,6,46,40},
{40,46,47,41},
{41,47,48,42},
{42,48,49,43},
{43,49,50,44},
{44,50,51,45},
{45,51,15,14},
{6,7,52,46},
{46,52,53,47},
{47,53,54,48},
{48,54,55,49},
{49,55,56,50},
{50,56,57,51},
{51,57,16,15},
{7,8,58,52},
{52,58,59,53},
{53,59,60,54},
{54,60,61,55},
{55,61,62,56},
{56,62,63,57},
{57,63,17,16},
{8,9,64,58},
{58,64,65,59},
{59,65,66,60},
{60,66,67,61},
{61,67,68,62},
{62,68,69,63},
{63,69,18,17},
{9,10,70,64},
{64,70,71,65},
{65,71,72,66},
{66,72,73,67},
{67,73,74,68},
{68,74,75,69},
{69,75,19,18},
{10,11,76,70},
{70,76,77,71},
{71,77,78,72},
{72,78,79,73},
{73,79,80,74},
{74,80,81,75},
{75,81,20,19},
{11,12,82,76},
{76,82,83,77},
{77,83,84,78},
{78,84,85,79},
{79,85,86,80},
{80,86,87,81},
{81,87,21,20},
{12,1,28,82},
{82,28,29,83},
{83,29,30,84},
{84,30,31,85},
{85,31,32,86},
{86,32,33,87},
{87,33,3,21}
};

//The initial partitions given by reading (simulating a read by blocks for large meshes):
const PetscInt sInitialPartition7x10Mesh[2][35] = {
  {0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38,42,43,44,48,49,50,54,55,56,60,61,62,66,67},
  {3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35,39,40,41,45,46,47,51,52,53,57,58,59,63,64,65,68,69}
};
const PetscInt sNLoclCells7x10Mesh = 35;
const PetscInt sNGlobVerts7x10Mesh = 88;


PetscReal sCoords2x5Mesh[18][2] = {
 {0.00000000000000000e+00, 0.00000000000000000e+00},
 {2.00000000000000000e+00, 0.00000000000000000e+00},
 {0.00000000000000000e+00, 1.00000000000000000e+00},
 {2.00000000000000000e+00, 1.00000000000000000e+00},
 {9.99999999997387978e-01, 0.00000000000000000e+00},
 {9.99999999997387978e-01, 1.00000000000000000e+00},
 {0.00000000000000000e+00, 2.00000000000000011e-01},
 {0.00000000000000000e+00, 4.00000000000000022e-01},
 {0.00000000000000000e+00, 5.99999999999999978e-01},
 {0.00000000000000000e+00, 8.00000000000000044e-01},
 {2.00000000000000000e+00, 2.00000000000000011e-01},
 {2.00000000000000000e+00, 4.00000000000000022e-01},
 {2.00000000000000000e+00, 5.99999999999999978e-01},
 {2.00000000000000000e+00, 8.00000000000000044e-01},
 {9.99999999997387756e-01, 2.00000000000000011e-01},
 {9.99999999997387978e-01, 4.00000000000000022e-01},
 {9.99999999997387978e-01, 6.00000000000000089e-01},
 {9.99999999997388089e-01, 8.00000000000000044e-01}};

//Connectivity of a 2x5 rectangular mesh of quads :
const PetscInt sConnectivity2x5Mesh[10][4] = {
  {0,4,14,6},
  {6,14,15,7},
  {7,15,16,8},
  {8,16,17,9},
  {9,17,5,2},
  {4,1,10,14},
  {14,10,11,15},
  {15,11,12,16},
  {16,12,13,17},
  {17,13,3,5}};

const PetscInt sInitialPartition2x5Mesh[2][5] = {
  {0,2,4,6,8},
  {1,3,5,7,9}
};

const PetscInt sNLoclCells2x5Mesh = 5;
const PetscInt sNGlobVerts2x5Mesh = 18;

int main(int argc, char **argv)
{
  const PetscInt   Nc                 = sNLoclCells2x5Mesh; //Same on each rank for this example...
  const PetscInt   Nv                 = sNGlobVerts2x5Mesh;
  const PetscInt*  InitPartForRank[2] = {&sInitialPartition2x5Mesh[0][0],
                                         &sInitialPartition2x5Mesh[1][0]};
  const PetscInt (*Conn)[4]           = sConnectivity2x5Mesh;

  const PetscInt   Ncor = 4;
  const PetscInt   dim  = 2;
  DM               dm, idm, ddm;
  PetscSF          sfVert, sfMig, sfPart;
  PetscPartitioner part;
  PetscSection     s;
  PetscInt        *cells, c;
  PetscMPIInt      size, rank;
  PetscBool        box = PETSC_FALSE, field = PETSC_FALSE;
  PetscErrorCode   ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  if (size != 2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a 2 processors example only");
  ierr = PetscOptionsGetBool(NULL, NULL, "-box", &box, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-field", &field, NULL);CHKERRQ(ierr);

  ierr = DMPlexCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  if (box) {
    ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(Nc * Ncor, &cells);CHKERRQ(ierr);
    for (c = 0; c < Nc; ++c) {
      PetscInt cell = (InitPartForRank[rank])[c], cor;

      for (cor = 0; cor < Ncor; ++cor) {
        cells[c*Ncor + cor] = Conn[cell][cor];
      }
    }
    ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
    ierr = DMPlexBuildFromCellListParallel(dm, Nc, PETSC_DECIDE, Nv, Ncor, cells, &sfVert);CHKERRQ(ierr);
    //ierr = DMPlexBuildCoordinatesFromCellListParallel(dm, dim, sfVert, coords);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfVert);CHKERRQ(ierr);
    ierr = PetscFree(cells);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = idm;
  }
  ierr = DMSetUseNatural(dm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  if (field) {
   const PetscInt Nf         = 1;
   const PetscInt numComp[1] = {1};
   const PetscInt numDof[3]  = {0, 0, 1};
   const PetscInt numBC      = 0;

   ierr = DMSetNumFields(dm, Nf);CHKERRQ(ierr);
   ierr = DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
   ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
   ierr = PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  }

  ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

  ierr = DMPlexDistribute(dm, 0, &sfMig, &ddm);CHKERRQ(ierr);
  ierr = PetscSFView(sfMig, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSFCreateInverseSF(sfMig, &sfPart);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) sfPart, "Inverse Migration SF");CHKERRQ(ierr);
  ierr = PetscSFView(sfPart, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  Vec          lGlobalVec, lNatVec;
  PetscScalar *lNatVecArray;

  {
    PetscSection s;

    ierr = DMGetGlobalSection(dm, &s);CHKERRQ(ierr);
    ierr = PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMGetGlobalVector(dm, &lNatVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) lNatVec, "Natural Vector (initial partition)");CHKERRQ(ierr);

  //Copying the initial partition into the "natural" vector:
  ierr = VecGetArray(lNatVec, &lNatVecArray);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) lNatVecArray[c] = (InitPartForRank[rank])[c];
  ierr = VecRestoreArray(lNatVec, &lNatVecArray);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(ddm,&lGlobalVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) lGlobalVec, "Global Vector (reordered element numbers in the petsc distributed order)");CHKERRQ(ierr);
  ierr = VecZeroEntries(lGlobalVec);CHKERRQ(ierr);

  // The call to DMPlexNaturalToGlobalBegin/End does not produce our expected result...
  // In lGlobalVec, we expect to have:
  /*
   * Process [0]
   * 2.
   * 4.
   * 8.
   * 3.
   * 9.
   * Process [1]
   * 1.
   * 5.
   * 7.
   * 0.
   * 6.
   *
   * but we obtained:
   *
   * Process [0]
   * 2.
   * 4.
   * 8.
   * 0.
   * 0.
   * Process [1]
   * 0.
   * 0.
   * 0.
   * 0.
   * 0.
   */

   {
     PetscSF nsf;

     ierr = DMPlexGetGlobalToNaturalSF(ddm, &nsf);CHKERRQ(ierr);
     ierr = PetscSFView(nsf, NULL);CHKERRQ(ierr);
   }
  ierr = DMPlexNaturalToGlobalBegin(ddm, lNatVec, lGlobalVec);CHKERRQ(ierr);
  ierr = DMPlexNaturalToGlobalEnd  (ddm, lNatVec, lGlobalVec);CHKERRQ(ierr);

  ierr = VecView(lNatVec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(lGlobalVec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dm,&lNatVec);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(ddm,&lGlobalVec);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&sfMig);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfPart);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&ddm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    args: -field
    nsize: 2

    test:
      suffix: 0
      args:

    test:
      suffix: 1
      args: -box -dm_plex_simplex 0 -dm_plex_box_faces 2,5 -dm_distribute

TEST*/
