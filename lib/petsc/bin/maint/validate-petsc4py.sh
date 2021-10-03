#!/bin/bash
# Check the (prefix) installation of petsc4py, try to import

set -e

die () {
	echo "ERROR: validate-petsc4py:" "$@" 1>&2
	exit 1
}

if [[ -z "${PYTHON}" ]]
then
	PYTHON=python
fi

[[ -n "${PETSC_DIR}" ]] || die "PETSC_DIR var not set"
# Subdirectory of PETSC_DIR/lib (pythonX.Y/site-packages) where petsc4py bindings get installed
[[ -n "${PY_DIR}" ]] || die "PY_DIR var not set"

PETSC_CFG="${PETSC_DIR}/lib/${PY_DIR}/petsc4py/lib/petsc.cfg"

[[ -f "${PETSC_CFG}" ]] || die "cfg file not found: ${PETSC_CFG}"
grep -q "^\s*PETSC_DIR\s*=\s*${PWD}/${T_PREFIX}\s*\$" "${PETSC_CFG}" || \
	die "invalid PETSC_DIR value in ${PETSC_CFG}"
grep -q "^\s*PETSC_ARCH\s*=\s*\$" "${PETSC_CFG}" || \
	die "invalid PETSC_ARCH value in ${PETSC_CFG}"

PYTHONPATH="${PETSC_DIR}/lib/${PY_DIR}:${PYTHONPATH}" \
	${PYTHON} -m petsc4py -help || \
	die "failed to import and invoke petsc4py"
