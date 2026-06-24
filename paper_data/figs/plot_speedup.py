"""
Figure 1: Block/scalar speedup ratio vs GPU count (weak scaling, 1 rank/GPU).
Data from Table 1, plans/baij-kokkos-paper.md (Phase-4, A100, ne=31/63/95/127).
Ratios = scalar_time / block_time (>1 means block is faster).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

gpus = [1, 8, 27, 64]

# scalar / block (> 1 → block wins; < 1 → scalar wins)
ksp  = [0.0428/0.0510, 0.1487/0.1428, 0.3380/0.2718, 0.3252/0.2793]
spmv = [0.0243/0.0361, 0.1003/0.0893, 0.2113/0.1492, 0.2125/0.1632]
ptap = [0.00595/0.00599, 0.01480/0.01021, 0.01889/0.01050, 0.02469/0.01088]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', zorder=0, label='parity (1×)')

ax.plot(gpus, ksp,  'o-',  color='#1f77b4', linewidth=1.8, markersize=6, label=r'KSPSolve')
ax.plot(gpus, spmv, 's--', color='#ff7f0e', linewidth=1.8, markersize=6, label=r'SpMV')
ax.plot(gpus, ptap, '^:',  color='#2ca02c', linewidth=1.8, markersize=6, label=r'PtAP')

# annotate each point with its ratio
for x, y in zip(gpus, ksp):
    ax.annotate(f'{y:.2f}×', xy=(x, y), xytext=(0, 7), textcoords='offset points',
                ha='center', fontsize=7.5, color='#1f77b4')
for x, y in zip(gpus, spmv):
    va = 'bottom' if y > ksp[gpus.index(x)] else 'top'
    ax.annotate(f'{y:.2f}×', xy=(x, y), xytext=(0, -14), textcoords='offset points',
                ha='center', fontsize=7.5, color='#ff7f0e')
for x, y in zip(gpus, ptap):
    ax.annotate(f'{y:.2f}×', xy=(x, y), xytext=(0, 7), textcoords='offset points',
                ha='center', fontsize=7.5, color='#2ca02c')

ax.set_xscale('log')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.set_xlabel('GPU count (1 rank/GPU)', fontsize=10)
ax.set_ylabel('Block / scalar speedup', fontsize=10)
ax.set_ylim(0.50, 2.65)
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# shade "block wins" region
ax.axhspan(1.0, 2.65, alpha=0.04, color='green', zorder=0)
ax.axhspan(0.50, 1.0, alpha=0.04, color='red', zorder=0)
ax.text(1.05, 1.04, 'block wins', fontsize=7.5, color='green', va='bottom')
ax.text(1.05, 0.96, 'scalar wins', fontsize=7.5, color='red', va='top')

ax.legend(loc='upper left', fontsize=8.5, frameon=True, framealpha=0.9)
ax.tick_params(axis='both', labelsize=9)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'speedup_vs_gpus.{ext}', bbox_inches='tight', dpi=300)
print('Wrote speedup_vs_gpus.pdf / .png')
