"""
Figure 3: Memory capacity — per-event bars at n=8 GPUs, ne=31/63/127.
ne=127: scalar OOMs; block fits.
ne=31/63: updated to 1-rank/GPU hot timings from Table 1.
  ne=31 → n=1 in Table 1 (single GPU, no MPI; ok to show as the small case)
  ne=63 → n=8 Table 1 numbers
  ne=127 → scalar OOM; block: rough estimate from 2-ranks/GPU run scaled down
           (scalar PtAP OOM, SpMV OOM, KSPSolve OOM → show "OOM" labels)
           block ne=127: KSPSolve~0.39, SpMV~0.28, PtAP~0.018 (rough from old data)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 1-rank/GPU data (Table 1) for ne=31 (n=1 GPU) and ne=63 (n=8 GPUs)
DATA = {
    31: {
        'block':  {'KSPSolve': 0.0510, 'SpMV': 0.0361, 'PtAP': 0.00599},
        'scalar': {'KSPSolve': 0.0428, 'SpMV': 0.0243, 'PtAP': 0.00595},
    },
    63: {
        'block':  {'KSPSolve': 0.1428, 'SpMV': 0.0893, 'PtAP': 0.01021},
        'scalar': {'KSPSolve': 0.1487, 'SpMV': 0.1003, 'PtAP': 0.01480},
    },
    127: {
        # 128^3 on 8 GPUs (8x overload, native-KK build). cuSPARSE spgemm OOMs (annotation);
        # native-KK scalar AND block both fit, 41 iters. KSPSolve no-timer; SpMV/PtAP gpu-timer.
        'block':  {'KSPSolve': 0.496, 'SpMV': 0.367, 'PtAP': 0.041},
        'scalar': {'KSPSolve': 0.387, 'SpMV': 0.292, 'PtAP': 0.325},
    },
}

events = ['KSPSolve', 'SpMV', 'PtAP']
nes = [31, 63, 127]
ne_labels = [r'$32^3$, 1 GPU', r'$64^3$, 8 GPUs', r'$128^3$, 8 GPUs']

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=False)
fig.suptitle(r'Block (BAIJ) vs scalar (AIJ): cuSPARSE runs out of memory at $128^3$ (native KK fits)',
             fontsize=11)

w = 0.38
x = np.arange(len(events))

for ax, ne, title in zip(axes, nes, ne_labels):
    bdata = DATA[ne]['block']
    sdata = DATA[ne]['scalar']
    bvals = [bdata[e] for e in events]
    svals = [sdata[e] for e in events]

    # block bars: value or n/a (not separately measured)
    for i, (e, bv) in enumerate(zip(events, bvals)):
        bx = x[i] - w/2
        if bv is None:
            ax.bar(bx, 0, w, color='#1f77b4', zorder=3,
                   label='block (BAIJ)' if i == 0 else '')
            ax.text(bx, max([v for v in bvals if v is not None], default=0.0) * 0.05,
                    'n/a', ha='center', va='bottom', fontsize=7, color='#1f77b4')
        else:
            ax.bar(bx, bv, w, color='#1f77b4', zorder=3,
                   label='block (BAIJ)' if i == 0 else '')
            ax.text(bx, bv * 1.03, f'{bv:.4f}' if bv < 0.1 else f'{bv:.3f}',
                    ha='center', va='bottom', fontsize=7)

    # scalar bars: OOM or value
    max_bval = max([v for v in bvals if v is not None], default=0.0)
    for i, (e, sv) in enumerate(zip(events, svals)):
        bx = x[i] + w/2
        if sv is None:
            oom_h = max_bval * 0.30
            ax.bar(bx, oom_h, w, facecolor='none', edgecolor='#d62728',
                   linestyle='--', linewidth=1.4, zorder=3,
                   label='scalar (AIJ)' if i == 0 else '')
            ax.text(bx, oom_h / 2, 'OOM', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#d62728', rotation=90)
        else:
            bar = ax.bar(bx, sv, w, color='#d62728', zorder=3,
                         label='scalar (AIJ)' if i == 0 else '')
            ax.text(bx, sv * 1.03,
                    f'{sv:.4f}' if sv < 0.1 else f'{sv:.3f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([r'KSPSolve', r'SpMV', r'PtAP'], fontsize=8.5)
    ax.set_title(title, fontsize=10)
    if ne == 127:
        ax.text(0.40, 0.90, 'cuSPARSE spgemm: OOM', transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, fontweight='bold', color='#d62728')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=8.5)
    if ne == 31:
        ax.set_ylabel('Time (s), hot', fontsize=9.5)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'events_oom.{ext}', bbox_inches='tight', dpi=300)
print('Wrote events_oom.pdf / .png')
