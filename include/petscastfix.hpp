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
void PetscValidIntPointer(T*,int);
template <typename T>
void PetscValidBoolPointer(T*,int);
template <typename T>
void PetscValidScalarPointer(T*,int);
template <typename T>
void PetscValidRealPointer(T*,int);

template <typename Ta,typename Tb>
void PetscCheckSameType(Ta,int,Tb,int);
template <typename T>
void PetscValidType(T,int);
template <typename Ta,typename Tb>
void PetscCheckSameComm(Ta,int,Tb,int);
template <typename Ta,typename Tb>
void PetscCheckSameTypeAndComm(Ta,int,Tb,int);

template <typename Ta,typename Tb>
void PetscValidLogicalCollectiveScalar(Ta,Tb,int);
template <typename Ta,typename Tb>
void PetscValidLogicalCollectiveReal(Ta,Tb,int);
template <typename Ta,typename Tb>
void PetscValidLogicalCollectiveInt(Ta,Tb,int);
template <typename Ta,typename Tb>
void PetscValidLogicalCollectiveMPIInt(Ta,Tb,int);
template <typename Ta,typename Tb>
void PetscValidLogicalCollectiveBool(Ta,Tb,int);
template <typename Ta,typename Tb>
void PetscValidLogicalCollectiveEnum(Ta,Tb,int);
#endif
