#!/bin/bash
#
#  Install petsc4py four ways and verify that each is installed correctly
set -e

function die () {
  echo "ERROR: validate-petsc4py-installs:" $@
  if [[ -d hide ]]; then
    mv hide/arch-* ./;
  fi
  rm -rf hide
  exit 1
}

if [[ -z "${PYTHON}" ]]; then
  export PYTHON=python3
fi

[[ -n "${PETSC_DIR}" ]] || die "PETSC_DIR variable not set"

export INITIALPYTHONPATH=${PYTHONPATH}

# Checks that petsc4py is loadable in the requested directory or default Python locations if no directory is provided
function checkpetsc4py() {
  PETSC4PY_PYTHONPATH=$1
  echo "Checking if petsc4py is properly installed"
  if [[ "x${PETSC4PY_PYTHONPATH}" != "x" ]]; then
    echo " in ${PETSC4PY_PYTHONPATH}"
    rm -rf hide
    mkdir hide
    mv arch-* hide/
    PETSC_CFG="${PETSC4PY_PYTHONPATH}/petsc4py/lib/petsc.cfg"
    [[ -f "${PETSC_CFG}" ]] || die "cfg file not found in ${PETSC_CFG}"
    grep -q "^\s*PETSC_DIR\s*=\s*${PWD}/${T_PREFIX}\s*\$" "${PETSC_CFG}" || die "invalid PETSC_DIR value in ${PETSC_CFG}"
    grep -q "^\s*PETSC_ARCH\s*=\s*\$" "${PETSC_CFG}" || die "invalid PETSC_ARCH value in ${PETSC_CFG}"
    export PYTHONPATH="${PETSC4PY_PYTHONPATH}:${INITIALPYTHONPATH}"
  fi
  ${PYTHON} -m petsc4py -help || die "failed to import and invoke petsc4py in PYTHONPATH " ${PYTHONPATH}
  if [[ "x${PETSC4PY_PYTHONPATH}" != "x" ]]; then
    export PYTHONPATH=${INITIALPYTHONPATH}
    mv hide/arch-* ./
    rm -rf hide
  fi
}

#  install from dir/* to /dir/*
function petscinstall {
  echo "Installing from DESTDIR to --prefix dir manually"
  find * -type d -exec install -d  "{}" "/{}" \; || die "failed petscinstall making directories" `pwd`
  find * -type f -exec install  "{}" "/{}" \; || die "failed petscinstall copying files" `pwd`
  find * -type l -exec install  "{}" "/{}" \; || die "failed petscinstall copying links" `pwd`
}

function main {
  T_PREFIX=petsc-install
  T_DESTDIR=petsc-destdir
  T_PETSC4PY=src/binding/petsc4py

  rm -rf "${PWD}/${T_PREFIX}" "${PWD}/${T_DESTDIR}"
  ${PYTHON} ./configure --prefix="${PWD}/${T_PREFIX}" --with-petsc4py=1 --with-debugging=0
  make CFLAGS=-Werror CXXFLAGS="-Werror -Wzero-as-null-pointer-constant" FFLAGS=-Werror


  printf "\n====== Test A. Install using --with-petsc4py into prefix directory with staging =====\n"
  make install DESTDIR="${PWD}/${T_DESTDIR}"
  test "$(find ${PWD}/${T_DESTDIR} -mindepth 1 | wc -l)" -gt 0
  (cd "${PWD}/${T_DESTDIR}" && petscinstall )
  checkpetsc4py `make getpetsc4pypythonpath`
  rm -rf "${PWD}/${T_PREFIX}" "${PWD}/${T_DESTDIR}"

  printf "\n====== Test B. Install using --with-petsc4py into prefix directory =====\n"
  make install
  test "$(find ${PWD}/${T_PREFIX} -mindepth 1 | wc -l)" -gt 0
  checkpetsc4py `make getpetsc4pypythonpath`
  PETSC4PY_PYTHONPATH=`make getpetsc4pypythonpath`

  printf "\n====== Test C. Install petscp4y manually with setuptools =====\n"
  export PETSC_DIR="${PWD}/${T_PREFIX}" && (cd "${T_PETSC4PY}" && ${PYTHON} setup.py build)
  export PETSC_DIR="${PWD}/${T_PREFIX}" P="${PWD}" && (cd "${T_PETSC4PY}" && ${PYTHON} setup.py install --install-lib="${PETSC4PY_PYTHONPATH}")
  checkpetsc4py ${PETSC4PY_PYTHONPATH}

  printf "\n====== Test D. Install petsc4py manually with setuptools with staging =====\n"
  export PETSC_DIR="${PWD}/${T_PREFIX}" P="${PWD}" && (cd "${T_PETSC4PY}" && ${PYTHON} setup.py install --root="${P}/${T_DESTDIR}" --install-lib="${PETSC4PY_PYTHONPATH}")
  (cd "${PWD}/${T_DESTDIR}" && petscinstall )
  checkpetsc4py  ${PETSC4PY_PYTHONPATH}
  rm -rf "${PWD}/${T_PREFIX}" "${PWD}/${T_DESTDIR}"
  export PETSC_DIR=`pwd`

  # requires pip less than 23.1
  printf "\n====== Test E. Install PETSc and petsc4py with pip in virtual environment; petsc is installed in $PETSC_DIR/$PETSC_ARCH while petsc4py is installed in virtenv system site-packages =====\n"
  ${PYTHON} -m venv pip-builds
  source pip-builds/bin/activate
  ${PYTHON} -m pip install --upgrade pip==23.0.1
  export PETSC_CONFIGURE_OPTIONS="--with-mpi=0 --with-fc=0"
  export CFLAGS="-O0"
  ${PYTHON} -m pip install .
  ${PYTHON} -m pip install src/binding/petsc4py
  checkpetsc4py
  deactivate
  unset PETSC_CONFIGURE_OPTIONS
  unset CFLAGS
  rm -rf  pip-builds

  # does not work with pip 23.1 or higher
  #printf "\n====== Test F. Install petsc is installed in $PETSC_DIR/$PETSC_ARCH while petsc4py is installed in $USER site-packages =====\n"
  #export PETSC_CONFIGURE_OPTIONS="--with-mpi=0 --with-fc=0"
  #export CFLAGS="-O0"
  #${PYTHON} -m pip install .
  #${PYTHON} -m pip install --user src/binding/petsc4py
  #checkpetsc4py ''
  #unset PETSC_CONFIGURE_OPTIONS
  #unset CFLAGS
}

main
