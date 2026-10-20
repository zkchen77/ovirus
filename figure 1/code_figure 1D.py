# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- 1. data
file_path = 'fig 1D.xlsx'
data = pd.read_excel(file_path)

sets = {}
for col in data.columns:
    vals = data[col].dropna().astype(str).str.strip()
    vals = vals[vals != '']
    sets[col] = set(vals)

set_names = list(sets.keys())
n_sets = len(set_names)
print("Set size:", {k: len(v) for k, v in sets.items()})

# --- 2. calculate
all_genes = set().union(*sets.values())
gene_membership = {}
for gene in all_genes:
    members = tuple(name for name in set_names if gene in sets[name])
    gene_membership[gene] = members

from collections import Counter
combo_counts = Counter(gene_membership.values())
combo_counts = {k: v for k, v in combo_counts.items() if len(k) > 0}
sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])

# ---3. draw UpSet
fig = plt.figure(figsize=(14, 9))

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1, 4],
    height_ratios=[3, 1.2],
    wspace=0.05, hspace=0.05
)

ax_bar = fig.add_subplot(gs[0, 1])
ax_size = fig.add_subplot(gs[1, 0])
ax_dot = fig.add_subplot(gs[1, 1])

n_combos = len(sorted_combos)
x_positions = range(n_combos)

counts = [c for _, c in sorted_combos]
ax_bar.bar(x_positions, counts, color='#4C72B0', width=0.6)
ax_bar.set_xlim(-0.5, n_combos - 0.5)
ax_bar.set_ylabel('Intersection size', fontsize=11)
ax_bar.set_xticks([])
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

for x, c in zip(x_positions, counts):
    ax_bar.text(x, c, str(c), ha='center', va='bottom', fontsize=18)

set_sizes = [len(sets[name]) for name in set_names]
y_positions = range(n_sets)
ax_size.barh(y_positions, set_sizes, color='#4C72B0', height=0.6)
ax_size.set_ylim(-0.5, n_sets - 0.5)
ax_size.invert_yaxis()
ax_size.set_yticks(y_positions)
ax_size.set_yticklabels(set_names, fontsize=18)
ax_size.set_xlabel('Set size', fontsize=18)
ax_size.spines['top'].set_visible(False)
ax_size.spines['right'].set_visible(False)
ax_size.invert_xaxis()

for y, s in zip(y_positions, set_sizes):
    ax_size.text(s, y, f' {s}', ha='left', va='center', fontsize=9)

for x, (combo, _) in enumerate(sorted_combos):
    active_rows = [set_names.index(name) for name in combo]
    inactive_rows = [i for i in range(n_sets) if i not in active_rows]

    ax_dot.scatter(
        [x] * len(inactive_rows), inactive_rows,
        s=80, color='lightgray', zorder=2
    )
    
    ax_dot.scatter(
        [x] * len(active_rows), active_rows,
        s=180, color='#4C72B0', zorder=3
    )

    if len(active_rows) > 1:
        ax_dot.plot(
            [x, x], [min(active_rows), max(active_rows)],
            color='#4C72B0', linewidth=2.5, zorder=2.5
        )

ax_dot.set_xlim(-0.5, n_combos - 0.5)
ax_dot.set_ylim(-0.5, n_sets - 0.5)
ax_dot.invert_yaxis()
ax_dot.set_xticks([])
ax_dot.set_yticks([])
ax_dot.spines['top'].set_visible(False)
ax_dot.spines['right'].set_visible(False)
ax_dot.spines['left'].set_visible(False)

plt.savefig('figure 1D.svg', dpi=300, bbox_inches='tight')
plt.savefig('figure 1D.pdf', dpi=300, bbox_inches='tight')
plt.show()