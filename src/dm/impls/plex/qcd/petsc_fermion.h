#pragma once
#include <petsc/private/dmpleximpl.h>
/****************************************************************************************************************
 * Need something better for the mass parameter and other action information.
 ****************************************************************************************************************
 */
typedef struct {
  PetscReal M;
} WilsonParameters;
typedef struct {
  PetscInt  Ls;
  PetscReal M5;
  PetscReal m;
} DomainWallParameters;

typedef struct {
  PetscInt  Ls;
  PetscReal c;
  PetscReal b;
  PetscReal M5;
  PetscReal m;
} MobiusDomainWallParameters;

// Must provide these in a single unit of compilation
extern WilsonParameters           TheWilsonParameters;
extern DomainWallParameters       TheDomainWallParameters;
extern MobiusDomainWallParameters TheMobiusDomainWallParameters;

// Old parameters
//const PetscReal M = 1.0; // M5
//const PetscReal m = 0.01; // mf
//const int Ls=8;

static WilsonParameters *GetWilsonParameters(void)
{
  return &TheWilsonParameters;
}
static void SetWilsonParameters(WilsonParameters *_p)
{
  TheWilsonParameters = *_p;
}
static DomainWallParameters *GetDomainWallParameters(void)
{
  return &TheDomainWallParameters;
}
static void SetDomainWallParameters(DomainWallParameters *_p)
{
  TheDomainWallParameters = *_p;
}
static MobiusDomainWallParameters *GetMobiusDomainWallParameters(void)
{
  return &TheMobiusDomainWallParameters;
}
static void SetMobiusDomainWallParameters(MobiusDomainWallParameters *_p)
{
  TheMobiusDomainWallParameters = *_p;
}
