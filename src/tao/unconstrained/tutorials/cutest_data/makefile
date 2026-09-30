include ${PETSC_DIR}/lib/petsc/conf/variables

ifeq ($(PETSC_PRECISION),single)
SIFPRECISION=sp
else ifeq ($(PETSC_PRECISION),double)
SIFPRECISION=dp
else ifeq ($(PETSC_PRECISION),__float128)
SIFPRECISION=qp
endif

SIFNAME=$(basename $(notdir $(SIF)))

.PHONY: all
all:
	${SIFDECODER} -$(SIFPRECISION) $(SIF)
	${FC} ${FC_FLAGS} ${FFLAGS} -c *.f
	${SL_LINKER} $(call SL_LINKER_FUNCTION,libcutest_$(SIFNAME),0) -o libcutest_$(SIFNAME).${SL_LINKER_SUFFIX} *.o ${PETSC_EXTERNAL_LIB_BASIC}
