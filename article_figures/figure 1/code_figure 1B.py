# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import pandas as pd
import matplotlib.pyplot as plt

def draw_forest_grouped_by_exposure(file_path, output_file, title="Effect Size (β)"):
    df = pd.read_csv(file_path)
    
    required_cols = ['exposure', 'outcome', 'b', 'se']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    df = df[required_cols].copy()
    df.rename(columns={'b': 'beta', 'se': 'se'}, inplace=True)
    df['lower'] = df['beta'] - 1.96 * df['se']
    df['upper'] = df['beta'] + 1.96 * df['se']
    exposure_order = pd.unique(df['exposure']) 
    df['exposure'] = pd.Categorical(df['exposure'], categories=exposure_order, ordered=True)
    df = df.sort_values(['exposure', 'beta'], ascending=[True, False]).reset_index(drop=True)

    df['label'] = df['exposure'].astype(str) + " → " + df['outcome'].astype(str)
    
    n = len(df)
    
    fixed_fig_width = 9 # 18.0
    fixed_fig_height = max(0.6 * n + 2, 6)
    fig = plt.figure(figsize=(fixed_fig_width, fixed_fig_height))

    ax = fig.add_axes([0.38, 0.08, 0.58, 0.88])
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)


    # 3. color
    colors = ['red' if b >= 0 else 'blue' for b in df['beta']]
    for idx in range(n):
        color = colors[idx]
        row = df.iloc[idx]
        ax.errorbar(
            x=row['beta'],
            y=idx,
            xerr=[[row['beta'] - row['lower']], [row['upper'] - row['beta']]],
            fmt='o',
            capsize=5,
            capthick=1,
            ecolor=color,
            markeredgecolor=color,
            markerfacecolor='white',
            markersize=6,
            color=color
        )

    wrapped_labels = [
        f"{exposure} →\n {outcome}"
        for exposure, outcome in zip(df['exposure'].astype(str), df['outcome'].astype(str))
        ]
    ax.set_yticks(range(n))
    ax.set_yticklabels(wrapped_labels, fontsize=16)

    # 4. exposure
    current_exposure = None
    group_start_idx = 0
    for idx, exposure in enumerate(df['exposure']):
        if current_exposure is None:
            current_exposure = exposure
            group_start_idx = idx
        elif exposure != current_exposure:
            ax.axhline(y=idx - 0.5, color='lightgray', linestyle='-', linewidth=1)
            current_exposure = exposure


    ax.set_xlabel('Effect Size (β) with 95% CI', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_title(title, fontsize=28, pad=20)
    ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.savefig(f"{output_file}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_forest_grouped_by_exposure(
        file_path='fig 1B.csv',
        output_file='figure 1B',
        title='Effect Size (β) of\nSkeletal Disorders on Infections'
    )

