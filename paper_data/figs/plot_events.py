#!/usr/bin/env python3
"""Grouped bar chart: HOT event time, block vs scalar Kokkos, one panel per ne.

Axes: x = event (KSPSolve, SpMV, PtAP), series = Mat type (block / scalar).
Data are A100 4xGPU n=8 HOT times in seconds, ex56 elasticity GAMG,
-ksp_norm_type unpreconditioned.

Timing source (deliberately mixed for accuracy):
  - KSPSolve  : run WITHOUT -log_view_gpu_time (3rd solve, stage 6). Per-event
                GPU sync inflates the end-to-end total, so the no-timer run gives
                the true wall time.
  - SpMV/PtAP : run WITH -log_view_gpu_time (MatMult stage 6; MatPtAPNumeric
                stage 3 = hot numeric-only re-setup). Async kernels are mis-timed
                without the GPU event timers, so these need the timers on.
scalar mpiaijkokkos OOMs at ne=127.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVENTS = ["KSPSolve", "SpMV", "PtAP"]

# time in seconds; None = not measured yet, "OOM" = out of memory
DATA = {
    31: {
        "block":  {"KSPSolve": 1.0417, "SpMV": 1.0250, "PtAP": 0.024954},
        "scalar": {"KSPSolve": 1.1597, "SpMV": 1.0807, "PtAP": 0.048284},
    },
    63: {
        "block":  {"KSPSolve": 1.2973, "SpMV": 1.1654, "PtAP": 0.13078},
        "scalar": {"KSPSolve": 1.3932, "SpMV": 1.2889, "PtAP": 0.074404},
    },
    127: {
        "block":  {"KSPSolve": 2.4417, "SpMV": 2.1590, "PtAP": 0.81755},
        "scalar": {"KSPSolve": "OOM", "SpMV": "OOM", "PtAP": "OOM"},
    },
}

BLOCK_COLOR = "#1f77b4"   # mpibaijkokkos
SCALAR_COLOR = "#d62728"  # mpiaijkokkos


def plot_ne(ne, ax=None):
    d = DATA[ne]
    x = np.arange(len(EVENTS))
    w = 0.38
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

    def bars(key, off, color, label):
        vals, oom = [], []
        for e in EVENTS:
            v = d[key][e]
            if v == "OOM":
                vals.append(0.0); oom.append(True)
            elif v is None:
                vals.append(0.0); oom.append(False)
            else:
                vals.append(v); oom.append(False)
        rects = ax.bar(x + off, vals, w, label=label, color=color)
        for r, v, o in zip(rects, vals, oom):
            if o:
                ax.text(r.get_x() + r.get_width() / 2, 0.02, "OOM", rotation=90,
                        ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
            elif v > 0:
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        return rects

    bars("block", -w / 2, BLOCK_COLOR, "block (mpibaijkokkos)")
    bars("scalar", w / 2, SCALAR_COLOR, "scalar (mpiaijkokkos)")

    ax.set_xticks(x)
    ax.set_xticklabels(EVENTS)
    ax.set_ylabel("HOT wall time (s), 4×A100 n=8")
    ax.set_title(f"ex56 elasticity GAMG, ne={ne}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

    if standalone:
        fig.tight_layout()
        out = f"/Users/markadams/Codes/petsc/plans/figs/events_ne{ne}.png"
        fig.savefig(out, dpi=150)
        print("wrote", out)
        return out


def plot_all(nes=(31, 63, 127)):
    fig, axes = plt.subplots(1, len(nes), figsize=(5.0 * len(nes), 4.3), sharey=True)
    for ax, ne in zip(np.atleast_1d(axes), nes):
        plot_ne(ne, ax=ax)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    out = "/Users/markadams/Codes/petsc/plans/figs/events_all.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args == ["all"]:
        plot_all()
    else:
        nes = [int(a) for a in args] or [63]
        for ne in nes:
            plot_ne(ne)
