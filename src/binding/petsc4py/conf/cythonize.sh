#!/bin/sh
topdir=$(cd $(dirname "$0")/.. && pwd)
python "$topdir/conf/cythonize.py" \
    --working "$topdir/src" $@ \
    "petsc4py/PETSc.pyx"
