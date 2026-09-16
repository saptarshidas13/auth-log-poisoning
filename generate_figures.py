"""Regenerate the four figures used in the paper and supplementary material from the
CSV files in results/. Run after the experiment scripts in src/experiments/ have
populated results/ (see README.md). Output is written to figures/.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

# ---- Fig. 4: robustness-utility tradeoff ----
df4 = pd.read_csv('results/p4_certificate.csv')
ut = df4[df4['analysis'] == 'utility_cost']
ut_uni = ut[ut['dataset'] == 'university'].copy()

fig, ax = plt.subplots(figsize=(4.4, 3.2))
strict = ut_uni[ut_uni['miner'] == 'strict_xu_stoller']
rhap = ut_uni[ut_uni['miner'] == 'rhapsody'].sort_values('tau')

ax.scatter(strict['under_privilege'], strict['over_privilege_after'], color='green', marker='*', s=180,
           label='Certified (strict)', zorder=5)
ax.plot(rhap['under_privilege'], rhap['over_privilege_after'], marker='o', color='#c0392b',
        linewidth=1.3, markersize=5, label=r'Relaxed ($\tau$ varies)')

offsets = [(5, 5), (5, -12), (-30, 8), (-30, -14)]
for (idx, row), off in zip(rhap.iterrows(), offsets):
    ax.annotate(rf'$\tau={row["tau"]}$', (row['under_privilege'], row['over_privilege_after']),
                fontsize=7, xytext=off, textcoords='offset points')

ax.set_xlabel('Under-privilege (legitimate requests denied)')
ax.set_ylabel('Over-privilege (illegitimate grants)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_utility_tradeoff.pdf')
plt.close()
print('saved fig_utility_tradeoff.pdf')

# ---- Fig. 3: RQ2, theory-predicted exploitability ----
d = pd.read_csv('results/p1_multi_target_sweep.csv')
genuine = d[~d['baseline_already_granted']].copy()
genuine['margin'] = genuine['ceiling_reliability'] - genuine['tau']
genuine['bucket'] = np.where(genuine['margin'] > 0, r'$\tau < $ ceiling' + '\n(theory: exploitable)',
                              r'$\tau \geq $ ceiling' + '\n(theory: not exploitable)')
rates = genuine.groupby('bucket')['success'].mean() * 100
counts = genuine.groupby('bucket')['success'].count()

fig, ax = plt.subplots(figsize=(4.0, 3.2))
bars = ax.bar(rates.index, rates.values, color=['#c0392b', '#2c5aa0'], width=0.55)
for bar, n in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f'n={n}',
            ha='center', fontsize=8)
ax.set_ylabel('Empirical exploit rate (%)')
ax.set_ylim(0, 85)
ax.grid(alpha=0.3, linewidth=0.5, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_rq2_validation.pdf')
plt.close()
print('saved fig_rq2_validation.pdf')

# ---- Fig. 5: coalition-composition ablation ----
dfc = pd.read_csv('results/p7_ablation_group_size.csv')
dfc = dfc[dfc['success'] == True]
summary = dfc.groupby('group_size')['poison_size'].agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(summary['group_size'], summary['mean'], marker='o', color='#2c5aa0', linewidth=1.5, markersize=6)
ax.set_xlabel('Colluding coalition size')
ax.set_ylabel(r'Minimal poison size $|\Delta^*|$')
ax.set_xticks(summary['group_size'])
ax.grid(alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_coalition_ablation.pdf')
plt.close()
print('saved fig_coalition_ablation.pdf')

# ---- Fig. 2: phase transition ----
df1 = pd.read_csv('results/p1_rule_generalization.csv')
fig, ax = plt.subplots(figsize=(4.6, 3.2))
markers = {'university': 'o', 'healthcare': 's', 'project-management': '^', 'workforce': 'D'}
for name, g in df1.groupby('dataset'):
    if name not in markers:
        continue
    g = g.sort_values('tau')
    succ = g[g['success'] == True]
    if len(succ) == 0:
        continue
    ax.plot(succ['tau'], succ['poison_size'], marker=markers[name], label=name, linewidth=1.3, markersize=5)
ax.set_xlabel(r'Reliability threshold $\tau$')
ax.set_ylabel(r'Poison set size $|\Delta|$')
ax.set_yscale('log')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_phase_transition.pdf')
plt.close()
print('saved fig_phase_transition.pdf')
