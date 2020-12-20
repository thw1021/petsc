include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

TARGET = eulerian3D
BINDIR = ../bin
OBJDIR = ../obj
TGT = $(BINDIR)/$(TARGET)

OBJECTS = petsc_wrapping/wrapper_petsc.o \
		  mod_common.o \
		  mod_input_parameter.o \
		  mod_geometry.o \
		  mod_riemannsolver.o \
		  mod_boundary.o \
		  mod_finitevolume.o \
		  mod_output.o \
		  mod_timestepper.o \
		  mod_solution.o \
		  main.o

.PHONY: build_tree

all: buildtree $(TGT) moveobjs

buildtree:
	-@mkdir -p $(OBJDIR)
	-@mkdir -p $(BINDIR)

moveobjs:
	-@mv *.o $(OBJDIR)/.
	-@mv *.mod $(OBJDIR)/.

$(TGT): $(OBJECTS)
	$(FC_LINKER) $(FC_LINKER_FLAGS) $(FFLAGS) $(PETSC_FCPPFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LOADLIBES) $(PETSC_LIB) -o $@
