"""
Figure 2: Per-event bar chart at n=8 GPUs, ne=63 (1 rank/GPU, Table 1 data).
block: KSPSolve=0.1428, SpMV=0.0893, PtAP=0.01021
scalar: KSPSolve=0.1487, SpMV=0.1003, PtAP=0.01480
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLOCK  = {'KSPSolve': 0.1428, 'SpMV': 0.0893, 'PtAP': 0.01021}
SCALAR = {'KSPSolve': 0.1487, 'SpMV': 0.1003, 'PtAP': 0.01480}
events = list(BLOCK.keys())

x = np.arange(len(events))
w = 0.38

fig, ax = plt.subplots(figsize=(5.0, 3.8))

bvals = [BLOCK[e]  for e in events]
svals = [SCALAR[e] for e in events]

rb = ax.bar(x - w/2, bvals, w, label='block (BAIJ)', color='#1f77b4', zorder=3)
rs = ax.bar(x + w/2, svals, w, label='scalar (AIJ)', color='#d62728', zorder=3)

for bar, val in zip(rb, bvals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)
for bar, val in zip(rs, svals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)

# speedup annotations
for i, e in enumerate(events):
    ratio = SCALAR[e] / BLOCK[e]
    top = max(BLOCK[e], SCALAR[e])
    ax.text(i, top + 0.006, f'{ratio:.2f}×', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color='black')

ax.set_xticks(x)
ax.set_xticklabels([r'KSPSolve', r'SpMV', r'PtAP'], fontsize=9.5)
ax.set_ylabel('Time (s), hot', fontsize=10)
ax.set_title(r'8 GPUs, $64^3$ grid, 1 rank/GPU', fontsize=10)
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9, frameon=True, framealpha=0.9)
ax.tick_params(axis='y', labelsize=9)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'events_n8.{ext}', bbox_inches='tight', dpi=300)
print('Wrote events_n8.pdf / .png')
