# -*- coding: utf-8 -*-
"""
Contour plot generation for LIQLEV-Python-parallel 2D geometry sweep.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import BASELINE_D, BASELINE_H, TARGET_RISE_IN


def plot_geometry_sweep(D_grid, H_grid, Rise_5s_grid, Rise_20s_grid,
                        Rise_Max_grid, Overfill_grid, vent_rate, output_dir):
    """
    Generate a 3-panel contour plot for a single vent rate.

    Parameters
    ----------
    D_grid, H_grid : ndarray
        Meshgrid arrays for diameter and height (feet).
    Rise_5s_grid : ndarray
        Level rise at t=5s (inches).
    Rise_20s_grid : ndarray
        Level rise at t=20s (inches).
    Rise_Max_grid : ndarray
        Maximum level rise (inches).
    Overfill_grid : ndarray
        Overfill flag (1.0 = tank overflowed, 0.0 = OK).
    vent_rate : float
        Vent rate (lbm/s) for title labeling.
    output_dir : str
        Directory to save the PNG plot.
    """
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150, sharey=True)

    plot_configs = [
        (axes[0], Rise_5s_grid, "Rise at t=5s"),
        (axes[1], Rise_20s_grid, "Rise at t=20s"),
        (axes[2], Rise_Max_grid, "Max Rise (Peak)"),
    ]

    vmin = 0
    vmax = np.max(Rise_Max_grid)

    for ax, data, title in plot_configs:
        cp = ax.contourf(D_grid, H_grid, data, levels=20,
                         cmap='viridis', vmin=vmin, vmax=vmax)

        # Hatched overlay for overfill regions
        ax.contourf(D_grid, H_grid, Overfill_grid,
                    levels=[0.5, 1.5], colors='none', hatches=['XX'])

        # Target rise contour line
        try:
            CS = ax.contour(D_grid, H_grid, data,
                            levels=[TARGET_RISE_IN], colors='red', linewidths=2)
            ax.clabel(CS, inline=True, fontsize=10,
                      fmt=f'{TARGET_RISE_IN:.1f}"')
        except (UserWarning, ValueError):
            pass

        # Baseline marker
        ax.plot(BASELINE_D, BASELINE_H, 'rx', markersize=10, markeredgewidth=2)

        ax.set_title(title)
        ax.set_xlabel('Diameter (ft)')
        if ax == axes[0]:
            ax.set_ylabel('Height (ft)')

    fig.colorbar(cp, ax=axes.ravel().tolist(), label='Rise (in)')

    # Legend
    hatch_patch = mpatches.Patch(facecolor='none', hatch='XX',
                                 label='Overfill (Max Reached)',
                                 edgecolor='black')
    line_patch = mpatches.Patch(color='red',
                                label=f'{TARGET_RISE_IN:.1f}" Target')
    fig.legend(handles=[hatch_patch, line_patch], loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.05))

    plt.suptitle(f'Level Rise - Flow: {vent_rate} lbm/s',
                 fontsize=16, y=1.1)

    filepath = os.path.join(output_dir, f"plots_flow_{vent_rate}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"   [*] Saved plot: {filepath}")
