#!/usr/bin/env bash
# Level-1 regression suite (guards self-improvement against drift).
#
# Rebuilds and reruns the verified components and re-validates the study/index
# artifacts against their schemas. A component promotion or skill edit is trusted
# only if this still passes. Requires PETSC_DIR/PETSC_ARCH set and mpiexec.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== [1/2] Component poisson2d-dmda: build, run, check 2nd-order convergence =="
( cd components/poisson2d-dmda
  make -s poisson
  mpiexec -n 4 ./poisson -mms_base 17 -mms_levels 4 -pc_type gamg -mms_csv convergence.csv
)
python3 - "$ROOT/components/poisson2d-dmda/convergence.csv" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def order(a, b, key):
    return math.log(float(a[key]) / float(b[key])) / math.log(float(a["h"]) / float(b["h"]))
oL2   = order(rows[-2], rows[-1], "L2")
oLinf = order(rows[-2], rows[-1], "Linf")
print(f"   observed order: L2={oL2:.3f}, Linf={oLinf:.3f} (expected ~2.0)")
sys.exit(0 if abs(oL2 - 2.0) < 0.1 and abs(oLinf - 2.0) < 0.1 else 1)
PY
echo "   [ ok ] convergence holds"

echo "== [2/2] Validate study + case-index artifacts against schemas =="
python3 tests/validate.py

echo "== REGRESSION PASSED =="
