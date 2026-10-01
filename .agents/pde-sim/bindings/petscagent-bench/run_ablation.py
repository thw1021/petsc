#!/usr/bin/env python3
"""Ablation driver: run petscagent-bench over the pde-sim Purple configs 1/2/3.

This drives the three-rung agent-configuration ablation described in
petscagent-bench's EXPERIMENT_PLAN.md:

  config 1  general coding agent          (one context, no skills, no subagents)
  config 2  general agent + skills        (one context, skills, no subagents)
  config 3  specialized + scoped skills   (full pde-sim pipeline)

All three are the SAME Purple binding (pdesim_purple.py) selected by PDESIM_CONFIG,
so the only thing that changes across rungs is the agent wiring under test.

Prerequisites (run this from the petscagent-bench repo root in its uv env so `src`
and the a2a deps import, e.g. ``cd ~/petscagent-bench && uv run python
~/petsc/.agents/pde-sim/bindings/petscagent-bench/run_ablation.py ...``):

  * The Green agent and the PETSc compile/run MCP server are ALREADY running
    (see the sibling README.md recipe). This driver only launches/tears down the
    Purple binding, one config at a time.
  * PETSC_DIR / PETSC_ARCH and the Green judge's LLM key are exported, exactly as
    for a normal bench run.

For each requested config it launches the Purple with PDESIM_CONFIG set, sends the
benchmark task to Green ``--repeats`` times (each send runs the whole data/*.json
suite and writes one output/<label>-judged-by-<green>-run<N>.json), tears the Purple
down, then moves on. Each send tags the task with ``<purple_model>`` = the config
label, so Green's own output filenames self-sort by config. Finally it aggregates
the per-config output JSONs into a comparison table (CSV + Markdown).

Use ``--aggregate-only`` to (re)build the report from existing output/*.json
without running anything.
"""

import argparse
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PURPLE_SCRIPT = Path(__file__).resolve().with_name("pdesim_purple.py")

# Same default Green AgentBeats ID as the bench's own client_cli.py.
DEFAULT_GREEN_ID = "019bb856-c8bf-7390-8c4f-bced52276932"


def _slug(name: str) -> str:
    """Filename-safe short form, matching green_agent.agent._slug()."""
    return re.sub(r"[^a-z0-9]+", "", (name or "unknown").lower().split("/")[-1])


def _name_slug(name: str) -> str:
    """Hyphen-preserving filename slug, matching green_agent.agent._name_slug();
    used to build/glob output filenames like pdesim-<model>-cN-judged-by-<green>."""
    return re.sub(r"[^a-z0-9-]+", "", (name or "unknown").lower().split("/")[-1]).strip("-") or "unknown"


def _task_text(purple_url: str, mcp_url: str, green_id: str, purple_id: str, purple_model: str) -> str:
    """Green task message, mirroring client_cli.py plus a <purple_model> label tag."""
    return (
        "\nYour task is to instantiate petscagent-bench to test the agent located at:\n"
        f"<purple_agent_url>\n{purple_url}/\n</purple_agent_url>\n"
        "You can use MCP tools from:\n"
        f"<mcp_server_url>\n{mcp_url}/\n</mcp_server_url>\n"
        f"<green_id>\n{green_id}\n</green_id>\n"
        f"<purple_id>\n{purple_id}\n</purple_id>\n"
        f"<purple_model>\n{purple_model}\n</purple_model>\n"
    )


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Block until (host, port) accepts a TCP connection, or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _launch_purple(config: str, host: str, port: int, claude_model: Optional[str],
                   cwd: Path, startup_timeout: float, usage_dir: Optional[str]) -> subprocess.Popen:
    """Start pdesim_purple.py with PDESIM_CONFIG set; wait until it is serving."""
    env = os.environ.copy()
    env["PDESIM_CONFIG"] = config
    if claude_model:
        env["PDESIM_CLAUDE_MODEL"] = claude_model
    if usage_dir:
        # A directory (absolute — the Purple runs with cwd=repo root, not the bench
        # dir); the Purple names the sidecar from its self-reported model, matching
        # the Green output file: <dir>/usage-pdesim-<model>-c<config>.jsonl.
        env["PDESIM_USAGE_DIR"] = usage_dir
    print(f"@@@ ablation: launching Purple config {config} on {host}:{port} "
          f"(model={claude_model or 'CLI default'})", flush=True)
    proc = subprocess.Popen(
        [sys.executable, str(PURPLE_SCRIPT), "--host", host, "--port", str(port)],
        env=env, cwd=str(cwd),
    )
    if not _wait_for_port(host, port, startup_timeout):
        proc.terminate()
        raise RuntimeError(f"Purple config {config} did not start within {startup_timeout:.0f}s")
    print(f"@@@ ablation: Purple config {config} ready", flush=True)
    return proc


def _stop_purple(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _config_label(config: str, args) -> str:
    """Purple-model label for a config: pdesim-<model>-c<N>.

    Must match what the binding self-reports in its telemetry (pdesim_purple.py
    _purple_model_label): the model slug of --claude-model (which the driver passes
    as PDESIM_CLAUDE_MODEL), or "clidefault" when unpinned. The binding self-reports
    this and Green names the output file from it; the driver reproduces the same
    label to locate those files and to tag the task as a fallback.
    """
    m = args.claude_model
    tag = _slug(m) if m else "clidefault"
    return f"pdesim-{tag}-c{config}"


def _usage_dir(args) -> Path:
    """Absolute output dir where the Purple writes per-config usage sidecars."""
    return (Path(args.bench_dir) / args.results_dir).resolve()


def _run_config(config: str, args, send_message) -> None:
    """Launch a config's Purple and fire `--repeats` benchmark runs at Green."""
    label = _config_label(config, args)
    usage_dir = _usage_dir(args)
    usage_dir.mkdir(parents=True, exist_ok=True)
    proc = _launch_purple(config, args.purple_host, args.purple_port, args.claude_model,
                          Path(args.bench_dir), args.startup_timeout, str(usage_dir))
    purple_url = f"http://{args.purple_host}:{args.purple_port}"
    try:
        for r in range(1, args.repeats + 1):
            print(f"@@@ ablation: config {config} repeat {r}/{args.repeats} "
                  f"(label={label}) -> {args.green_url}", flush=True)
            text = _task_text(purple_url, args.mcp_url, args.green_id, args.purple_id, label)
            response = asyncio.run(send_message(args.green_url, text))
            print(f"@@@ ablation: config {config} repeat {r} done; Green response tail:\n"
                  f"{str(response)[-500:]}", flush=True)
    finally:
        _stop_purple(proc)
        print(f"@@@ ablation: Purple config {config} stopped", flush=True)


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _aggregate(configs: List[str], args) -> List[Dict]:
    """Summarize each config's output/*.json files into one row per config."""
    output_dir = Path(args.results_dir)
    judge = _name_slug(args.judge_model) if args.judge_model else None
    rows: List[Dict] = []
    for config in configs:
        label = _config_label(config, args)
        # The binding names the file from the ACTUAL model it ran, with hyphens kept
        # (pdesim-<model>-c<N>). When pinned that equals our label; when unpinned we
        # can't predict the model, so wildcard just that part.
        prefix = _name_slug(label) if args.claude_model else f"pdesim-*-c{config}"
        pattern = f"{prefix}-judged-by-{judge}-run*.json" if judge else f"{prefix}-judged-by-*-run*.json"
        files = sorted(output_dir.glob(pattern))
        composites: List[float] = []
        compile_flags: List[float] = []
        run_flags: List[float] = []
        times: List[float] = []
        tokens = 0
        n_problems: List[int] = []
        tier_counts: Dict[str, int] = {}
        for f in files:
            data = json.loads(f.read_text())
            results = data.get("results", [])
            n_problems.append(len(results))
            for br in results:
                composites.append(br.get("composite_score"))
                compile_flags.append(1.0 if br.get("compiles") else 0.0)
                run_flags.append(1.0 if br.get("runs") else 0.0)
                times.append(br.get("time_used_sec"))
                tier = br.get("tier")
                if tier:
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
            tokens += (data.get("summary", {}) or {}).get("total_tokens", 0) or 0

        # Generation cost/turns from the per-config usage sidecar (config 3's cost
        # includes subagents via total_cost_usd; the bench token columns above do
        # not, so this is the trustworthy generation-cost signal).
        gen_cost = 0.0
        gen_records = 0
        subagents = 0
        skill_uses = 0
        skills_seen: set = set()
        actual_model: Optional[str] = None
        # Sidecars are named by the binding from the actual model, mirroring the
        # output-file prefix (wildcard the model when unpinned).
        usage_prefix = _name_slug(_config_label(config, args)) if args.claude_model else f"pdesim-*-c{config}"
        for usage_file in sorted(output_dir.glob(f"usage-{usage_prefix}.jsonl")):
            for line in usage_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gen_records += 1
                gen_cost += rec.get("total_cost_usd") or 0.0
                subagents += rec.get("subagents_spawned") or 0
                read = rec.get("pdesim_skills_read") or []
                skill_uses += (rec.get("skill_tool_calls") or 0) + len(read)
                skills_seen.update(read)
                actual_model = actual_model or rec.get("model_name")

        # Report the label built from the ACTUAL model the binding ran, when known,
        # so the row matches the real output filename even for unpinned runs.
        display_label = f"pdesim-{_slug(actual_model)}-c{config}" if actual_model else label

        rows.append({
            "config": config,
            "label": display_label,
            "n_runs": len(files),
            "n_problems_per_run": n_problems[0] if n_problems else 0,
            "avg_composite": _mean(composites),
            "compile_rate": _mean(compile_flags),
            "run_rate": _mean(run_flags),
            "avg_time_sec": _mean(times),
            "total_tokens": tokens,
            "skill_uses": skill_uses,
            "skills_seen": sorted(skills_seen),
            "gen_cost_usd": gen_cost if gen_records else None,
            "subagents": subagents,
            "tiers": tier_counts,
        })
    return rows


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _write_report(rows: List[Dict], args) -> None:
    csv_path = Path(args.results_dir) / "ablation_report.csv"
    md_path = Path(args.report_out)
    header = ["config", "label", "n_runs", "n_problems_per_run", "avg_composite",
              "compile_rate", "run_rate", "avg_time_sec", "total_tokens",
              "gen_cost_usd", "subagents", "skill_uses", "skills_seen"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            r["config"], r["label"], str(r["n_runs"]), str(r["n_problems_per_run"]),
            _fmt(r["avg_composite"]), _fmt(r["compile_rate"]), _fmt(r["run_rate"]),
            _fmt(r["avg_time_sec"], 2), str(r["total_tokens"]),
            _fmt(r["gen_cost_usd"], 4), str(r["subagents"]),
            str(r["skill_uses"]), "|".join(r["skills_seen"]),
        ]))
    csv_path.write_text("\n".join(lines) + "\n")

    md = ["# pde-sim ablation results",
          "",
          f"Judge model: `{args.judge_model or 'unspecified'}`  ·  "
          f"generator: `{args.claude_model or 'CLI default'}`",
          "",
          "Tokens = bench-reported (top-level agent). Gen $ = generation cost from the "
          "usage sidecar (subagent-inclusive `total_cost_usd`). Subagents / Skill uses = "
          "observed evidence from the transcript (skill-tool calls + SKILL.md reads).",
          "",
          "| Config | Label | Runs | Probs/run | Avg composite | Compile rate | Run rate | Avg time (s) | Tokens | Gen $ | Subagents | Skill uses |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"1": "general", "2": "single-context pipeline", "3": "specialized+scoped"}
    for r in rows:
        md.append(f"| {r['config']} ({names.get(r['config'], '')}) | {r['label']} | "
                  f"{r['n_runs']} | {r['n_problems_per_run']} | {_fmt(r['avg_composite'])} | "
                  f"{_fmt(r['compile_rate'])} | {_fmt(r['run_rate'])} | "
                  f"{_fmt(r['avg_time_sec'], 2)} | {r['total_tokens']} | "
                  f"{_fmt(r['gen_cost_usd'], 4)} | {r['subagents']} | {r['skill_uses']} |")
    md += ["", "Ladder deltas (avg composite):"]
    by_cfg = {r["config"]: r["avg_composite"] for r in rows}
    for lo, hi, what in (("1", "2", "skills"), ("2", "3", "specialization")):
        a, b = by_cfg.get(lo), by_cfg.get(hi)
        delta = f"{b - a:+.3f}" if (a is not None and b is not None) else "-"
        md.append(f"- {lo}→{hi} ({what}): {delta}")
    md.append("")
    md_path.write_text("\n".join(md) + "\n")
    print(f"@@@ ablation: wrote {csv_path} and {md_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default="1,2,3", help="comma-separated configs to run (default 1,2,3)")
    p.add_argument("--repeats", type=int, default=3, help="benchmark runs per config (default 3)")
    p.add_argument("--green-url", default="http://localhost:9001")
    p.add_argument("--mcp-url", default="http://localhost:8080/mcp")
    p.add_argument("--green-id", default=DEFAULT_GREEN_ID)
    p.add_argument("--purple-id", default="")
    p.add_argument("--purple-host", default="localhost")
    p.add_argument("--purple-port", type=int, default=9002)
    p.add_argument("--claude-model", default=None,
                   help="PDESIM_CLAUDE_MODEL for the generator; also forms the self-reported "
                        "purple_model label pdesim-<model>-cN (pdesim-clidefault-cN when unset)")
    p.add_argument("--judge-model", default="anthropic/claudeopus46",
                   help="judge slug used to locate output files (must match green_agent_config.yaml)")
    p.add_argument("--bench-dir", default=str(Path.cwd()), help="petscagent-bench repo root (cwd of the Purple)")
    p.add_argument("--results-dir", default="output", help="where Green writes result JSONs")
    p.add_argument("--report-out", default="output/ablation_report.md")
    p.add_argument("--startup-timeout", type=float, default=60.0)
    p.add_argument("--aggregate-only", action="store_true", help="only rebuild the report from existing outputs")
    args = p.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in ("1", "2", "3"):
            p.error(f"unknown config {c!r}; choose from 1,2,3")

    if not args.aggregate_only:
        # Import here so --aggregate-only works outside the bench env.
        from src.util.a2a_comm import send_message  # type: ignore
        for config in configs:
            _run_config(config, args, send_message)

    rows = _aggregate(configs, args)
    _write_report(rows, args)


if __name__ == "__main__":
    main()
