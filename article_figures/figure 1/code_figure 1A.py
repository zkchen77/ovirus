# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

df = pd.read_csv('fig 1A.csv', sep=',')

alpha = 0.05

exposures = sorted(df['Exposure'].unique())
outcomes = sorted(df['Outcome'].unique())

# --- heatmap_data：index=outcomes, columns=exposures
heatmap_data = pd.DataFrame(index=outcomes, columns=exposures, dtype=object)
heatmap_data[:] = '#f4f5f6'  # 全部初始化为 None 颜色

# --- init
for _, row in df.iterrows():
    exp = row['Exposure']
    out = row['Outcome']
    
    p_mr = row['pval Forward MR']
    p_reverse = row['pval Reverse MR']

    sig_mr = not pd.isna(p_mr) and p_mr < alpha
    sig_reverse = not pd.isna(p_reverse) and p_reverse < alpha

    if sig_mr and sig_reverse:
        color_code = '#fced69'
    elif sig_mr and not sig_reverse:
        color_code = '#fc6977'
    elif not sig_mr and sig_reverse:
        color_code = '#69c1fc'
    else:
        color_code = '#f4f5f6'

    heatmap_data.loc[out, exp] = color_code

# --- color map
colors = ['#f4f5f6', '#fc6977', '#69c1fc', '#fced69']
color_to_num = {col: i for i, col in enumerate(colors)}

# to matrix
numeric_data = heatmap_data.replace(color_to_num).astype(int)
numeric_data_transposed = numeric_data.T

# ListedColormap 
cmap = ListedColormap(colors)
vmin, vmax = 0, 3

# draw pic
plt.figure(figsize=(21, 7))
ax = sns.heatmap(
    numeric_data_transposed,
    cmap=cmap,
    cbar=False,
    linewidths=0.8,
    linecolor='black',
    square=True,
    xticklabels=True,
    yticklabels=True,
    vmin=vmin,
    vmax=vmax
)


for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

plt.xticks(rotation=45, ha='right', va='top', fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.xlabel('Outcome', fontsize=20)
plt.ylabel('Exposure', fontsize=20)
plt.title('Mendelian randomization bidirectional effects', fontsize=28, pad=20)

legend_elements = [
    Patch(facecolor='#fc6977', label='Forward'),
    Patch(facecolor='#69c1fc', label='Reverse'),
    Patch(facecolor='#fced69', label='Bidirectional'),
    Patch(facecolor='#f4f5f6', label='None')
]
plt.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(1.3, 1.0),
    title='bidirectional effects',
    frameon=True,
    fontsize=14,
    title_fontsize=14
)


plt.subplots_adjust(right=0.7)
plt.tight_layout()

plt.savefig('figure 1A.svg', dpi=300, bbox_inches='tight')
plt.savefig('figure 1A.pdf', dpi=300, bbox_inches='tight')
plt.show()
