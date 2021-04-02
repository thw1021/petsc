#ifndef PETSCASTFIX_HPP
#define PETSCASTFIX_HPP

#ifndef PETSC_AST_FIX
#define PETSC_AST_FIX
#endif

#include <petscsystypes.h>
template <typename T>
void PetscValidHeaderSpecific(const T,PetscClassId,int);
template <typename T>
void PetscValidHeaderSpecific(T,PetscClassId,int);
template <typename T>
void PetscValidHeaderSpecificType(T,PetscClassId,int,const char[]);
template <typename T>
void PetscValidHeader(T,int);
template <typename T>
void PetscValidPointer(T,int);
template <typename T>
void PetscValidCharPointer(T*,int);
template <typename T>
void PetscValidCharPointer(const T*,int);
template <typename T>
void PetscValidCharPointer(const T[],int);
template <typename T>
void PetscValidIntPointer(T*,int);
template <typename T>
void PetscValidBoolPointer(T*,int);
template <typename T>
void PetscValidScalarPointer(T*,int);
template <typename T>
void PetscValidRealPointer(T*,int);
template <typename T>
void PetscValidFunction(T*,int);
#endif
