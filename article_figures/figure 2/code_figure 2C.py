# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

# === config
xlsx_path = r'fig 2C.xlsx'
txt_folder = r'immport'
symbol_col_name = 'Symbol'

# 1. data
df_modes = pd.read_excel(xlsx_path, sheet_name='Sheet1')

mode1 = set(df_modes['Pattern I'].dropna().astype(str))
mode2 = set(df_modes['Pattern II'].dropna().astype(str))
mode3 = set(df_modes['Pattern III'].dropna().astype(str))

all_genes_ordered = []
seen = set()
for col in [df_modes['Pattern I'], df_modes['Pattern II'], df_modes['Pattern III']]:
    for gene in col.dropna().astype(str):
        if gene not in seen:
            all_genes_ordered.append(gene)
            seen.add(gene)

genes = all_genes_ordered

mode_labels = []
for gene in genes:
    if gene in mode1:
        mode_labels.append('Pattern I')
    elif gene in mode2:
        mode_labels.append('Pattern II')
    else:
        mode_labels.append('Pattern III')

# 2. Symbols
raw_txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
raw_txt_files.sort()
display_names = [os.path.splitext(f)[0] for f in raw_txt_files]

# 3. calculate matrix
matrix = []
for txt_file in raw_txt_files:
    file_path = os.path.join(txt_folder, txt_file)
    try:
        df_txt = pd.read_csv(file_path, sep='\t')
    except:
        try:
            df_txt = pd.read_csv(file_path, sep=',')
        except:
            print(f"error: {txt_file}")
            matrix.append([False] * len(genes))
            continue

    if symbol_col_name not in df_txt.columns:
        print(f"warning: file {txt_file} have no '{symbol_col_name}' cols")
        matrix.append([False] * len(genes))
        continue

    symbols_in_file = set(df_txt[symbol_col_name].dropna().astype(str))
    row = [gene in symbols_in_file for gene in genes]
    matrix.append(row)

heatmap_df = pd.DataFrame(matrix, index=display_names, columns=genes)

# 4. colors
color_map = {
    'Pattern I': '#add8e6',  # light blue
    'Pattern II': '#f8bfbf',  # light red
    'Pattern III': '#fffacd'   # light yellow
}
col_colors = [color_map[label] for label in mode_labels]

# 5. x-aixs
x_tick_colors = []
for gene in genes:
    in_m1 = gene in mode1
    in_m2 = gene in mode2
    in_m3 = gene in mode3

    if in_m1 and in_m2 and in_m3:
        color = 'purple'
    elif in_m1 and in_m2:
        color = 'red'
    elif in_m1 and in_m3:
        color = 'blue'
    elif in_m2 and in_m3:
        color = 'orange'
    else:
        color = 'black'
    x_tick_colors.append(color)

# 6. draw picture
plt.figure(figsize=(30, max(8, len(display_names) * 0.35)))
ax = sns.heatmap(
    heatmap_df,
    cmap=['white', 'steelblue'],
    cbar=False,
    linewidths=0.5,
    linecolor='lightgray',
    xticklabels=False
)

ax.tick_params(axis='y', labelsize=18)

for i, color in enumerate(col_colors):
    ax.add_patch(plt.Rectangle((i, -0.5), 1, 0.5, facecolor=color, edgecolor='none', clip_on=False))

ax.set_xticks(np.arange(len(genes)) + 0.5)
ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=18)
for tick_label, color in zip(ax.get_xticklabels(), x_tick_colors):
    tick_label.set_color(color)

legend_patches = [
    mpatches.Patch(color='#add8e6', label='Pattern I'),
    mpatches.Patch(color='#f8bfbf', label='Pattern II'),
    mpatches.Patch(color='#fffacd', label='Pattern III')
]
plt.legend(handles=legend_patches, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.17), 
           ncol=3, 
           fontsize=18,
           frameon=True)

plt.title('Gene Presence in ImmPort pathways', fontsize=24, pad=70)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig("figure 2C.svg", bbox_inches='tight')
plt.savefig("figure 2C.pdf", bbox_inches='tight')
plt.show()