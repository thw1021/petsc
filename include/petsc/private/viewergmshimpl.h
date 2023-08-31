#ifndef viewergmshimpl_h
#define viewergmshimpl_h

#include <petsc/private/viewerimpl.h>

typedef struct {
  PetscViewer   viewer;
  char         *filename;
  PetscFileMode btype;
  int           fileFormat;
  int           dataSize;
  PetscBool     binary;
  PetscBool     byteSwap;
  size_t        wlen;
  void         *wbuf;
  size_t        slen;
  void         *sbuf;
  PetscInt     *nbuf;
  PetscInt      nodeStart;
  PetscInt      nodeEnd;
  PetscInt     *nodeMap;
} PetscViewer_GMSH;

#endif
