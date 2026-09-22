#!/usr/bin/env python3
"""Dry-run validator: check each pipeline artifact against its contract schema,
verify cross-references, and independently recompute convergence orders.

Discovers every worked study under examples/ (any directory holding a
problem-spec.json) and always validates components/case-index.json, so it stays
correct as studies are added or removed."""
import json, sys, math, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "contracts"
EXAMPLES = ROOT / "examples"

PAIRS = [
    ("problem-spec.json",        "problem-spec.schema.json"),
    ("numerical-plan.json",      "numerical-plan.schema.json"),
    ("vis-spec.json",            "vis-spec.schema.json"),
    ("results-manifest.json",    "results-manifest.schema.json"),
    ("numerical-assessment.json","numerical-assessment.schema.json"),
    ("analysis-report.json",     "analysis-report.schema.json"),
]

def load(p): return json.loads(pathlib.Path(p).read_text())

def jtype_ok(v, t):
    if t == "object":  return isinstance(v, dict)
    if t == "array":   return isinstance(v, list)
    if t == "string":  return isinstance(v, str)
    if t == "boolean": return isinstance(v, bool)
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":  return isinstance(v, (int, float)) and not isinstance(v, bool)
    return True

def validate(inst, schema, path, errs):
    """Recursive Draft-2020-12 subset validator."""
    t = schema.get("type")
    if t and not jtype_ok(inst, t):
        errs.append(f"{path or '<root>'}: expected type {t}, got {type(inst).__name__}")
        return
    if "const" in schema and inst != schema["const"]:
        errs.append(f"{path}: expected const {schema['const']!r}, got {inst!r}")
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in enum {schema['enum']}")
    if isinstance(inst, dict):
        for req in schema.get("required", []):
            if req not in inst:
                errs.append(f"{path}: missing required '{req}'")
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        for k, v in inst.items():
            kp = f"{path}.{k}" if path else k
            if k in props:
                validate(v, props[k], kp, errs)
            elif addl is False:
                errs.append(f"{kp}: additional property not allowed")
            elif isinstance(addl, dict):
                validate(v, addl, kp, errs)
    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: needs >= {schema['minItems']} items, got {len(inst)}")
        if "items" in schema:
            for i, el in enumerate(inst):
                validate(el, schema["items"], f"{path}[{i}]", errs)

def check_study(study):
    """Validate one worked study; return True if everything passed."""
    ok = True
    print(f"\n### study: {study.name}")
    print("--- schema validation ---")
    for artifact, schema in PAIRS:
        a = load(study / artifact)
        s = load(CONTRACTS / schema)
        errs = []
        validate(a, s, "", errs)
        if errs:
            ok = False
            print(f"[FAIL] {artifact}")
            for e in errs[:12]:
                print(f"       {e}")
        else:
            print(f"[ ok ] {artifact}")

    print("--- cross-reference integrity ---")
    ps  = load(study / "problem-spec.json")
    np_ = load(study / "numerical-plan.json")
    vs  = load(study / "vis-spec.json")
    rm  = load(study / "results-manifest.json")
    na  = load(study / "numerical-assessment.json")
    ar  = load(study / "analysis-report.json")
    checks = [
        ("numerical-plan.problem_spec_id -> problem-spec.id", np_["problem_spec_id"] == ps["id"]),
        ("vis-spec.problem_spec_id -> problem-spec.id",       vs["problem_spec_id"] == ps["id"]),
        ("vis-spec.numerical_plan_id -> numerical-plan.id",   vs["numerical_plan_id"] == np_["id"]),
        ("results.numerical_plan_id -> numerical-plan.id",    rm["numerical_plan_id"] == np_["id"]),
        ("results.vis_spec_id -> vis-spec.id",                rm["vis_spec_id"] == vs["id"]),
        ("assessment.results_manifest_id -> results.id",      na["results_manifest_id"] == rm["id"]),
        ("analysis.results_manifest_id -> results.id",        ar["results_manifest_id"] == rm["id"]),
    ]
    for name, good in checks:
        print(f"[{'ok' if good else 'FAIL'}] {name}")
        ok = ok and good

    print("--- independent convergence-order recomputation ---")
    levels = rm["convergence_study"]["levels"]
    for norm in ("L2", "Linf"):
        print(f"  {norm}:")
        orders = []
        for i in range(1, len(levels)):
            e0, e1 = levels[i-1]["errors"][norm], levels[i]["errors"][norm]
            h0, h1 = levels[i-1]["h"], levels[i]["h"]
            p = math.log(e0/e1) / math.log(h0/h1)
            orders.append(p)
            print(f"    h {h0:.5f}->{h1:.5f}: order = {p:.3f}")
        avg = sum(orders)/len(orders)
        claimed = next(c["observed_order"] for c in na["convergence"] if c["norm"] == norm)
        good = abs(avg - claimed) < 0.05
        print(f"    mean {avg:.3f} vs assessment claim {claimed} -> [{'ok' if good else 'FAIL'}]")
        ok = ok and good
    return ok

print("using: self-contained subset validator")
ok = True

# case index (reuse registry) validates against its own schema
ci = load(ROOT / "components" / "case-index.json")
cis = load(CONTRACTS / "case-index.schema.json")
cerrs = []
validate(ci, cis, "", cerrs)
if cerrs:
    ok = False
    print("[FAIL] case-index.json")
    for e in cerrs[:12]:
        print(f"       {e}")
else:
    print(f"[ ok ] case-index.json ({len(ci['cases'])} case(s))")

# every worked study under examples/ (a directory holding a problem-spec.json)
studies = sorted(d for d in EXAMPLES.glob("*") if (d / "problem-spec.json").is_file())
if not studies:
    print("\nno worked studies under examples/ to validate")
for study in studies:
    ok = check_study(study) and ok

print("\n=== DRY-RUN", "PASSED ===" if ok else "FAILED ===")
sys.exit(0 if ok else 1)
