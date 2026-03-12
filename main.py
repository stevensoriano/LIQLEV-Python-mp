# -*- coding: utf-8 -*-
"""
LIQLEV-Python-parallel: Main execution script.
2D geometry sweep with multiprocessing.

Runs LIQLEV simulations in parallel across a diameter x height grid
for each vent rate. Exports results to Excel and generates contour plots.
"""
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    get_config, get_base_inputs, get_max_workers,
    VENT_RATES, SWEEP_FILL, D_RANGE, H_RANGE,
    OUTPUT_DIR, SNAPSHOT_5S, SNAPSHOT_20S,
)
from core import liqlev_simulation
from plotting import plot_geometry_sweep


def run_single_simulation_case(params):
    """
    Worker function for a single (diameter, height) geometry case.

    Parameters
    ----------
    params : tuple
        (i, j, d_val, h_val, vent_rate, fill_fraction)

    Returns
    -------
    tuple
        (i, j, rise_5s, rise_20s, rise_max, is_overfill, error)
    """
    i, j, d_val, h_val, vent_rate, fill = params

    try:
        cfg = get_config(dtank_ft=d_val, tank_height_ft=h_val)
        inps = get_base_inputs(cfg, vent_rate, fill, neps=0)
        df = liqlev_simulation(inps)

        if not df.empty:
            time_arr = df['Time'].values
            height_arr = df['Height'].values
            initial_h = height_arr[0]

            # Rise at 5s snapshot
            h_5s = np.interp(SNAPSHOT_5S, time_arr, height_arr)
            rise_5s = (h_5s - initial_h) * 12.0

            # Rise at 20s snapshot (or end of simulation)
            h_20s = np.interp(SNAPSHOT_20S, time_arr, height_arr)
            rise_20s = (h_20s - initial_h) * 12.0

            # Maximum rise
            max_h = np.max(height_arr)
            rise_max = (max_h - initial_h) * 12.0

            # Overfill check (99.9% threshold for float tolerance)
            is_overfill = 1.0 if max_h >= (h_val * 0.9999) else 0.0

            return (i, j, rise_5s, rise_20s, rise_max, is_overfill, None)
        else:
            return (i, j, 0.0, 0.0, 0.0, 0.0, "Empty Result")

    except Exception as e:
        return (i, j, 0.0, 0.0, 0.0, 0.0, str(e))


if __name__ == "__main__":

    start_time = time.time()

    # Resolve worker count
    max_workers = get_max_workers()
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    print(f"Running with {max_workers} workers on "
          f"{slurm_cpus if slurm_cpus else 'local'} CPUs.")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    d_range = D_RANGE
    h_range = H_RANGE

    print("\n=== STARTING 2D GEOMETRY SWEEP (MULTI-FLOW RATE) ===")

    for vent_rate in VENT_RATES:
        print(f"\n>>> Processing Flow Rate: {vent_rate} lbm/s")

        # Initialize grids
        D_grid, H_grid = np.meshgrid(d_range, h_range)
        Rise_5s_grid = np.zeros_like(D_grid)
        Rise_20s_grid = np.zeros_like(D_grid)
        Rise_Max_grid = np.zeros_like(D_grid)
        Overfill_grid = np.zeros_like(D_grid)

        # Build task list
        tasks = []
        for i in range(len(h_range)):
            for j in range(len(d_range)):
                tasks.append((i, j, d_range[j], h_range[i],
                              vent_rate, SWEEP_FILL))
        total_sims = len(tasks)
        completed_count = 0

        print(f"   Launching {total_sims} simulations...")

        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_simulation_case, task): task
                       for task in tasks}

            for future in as_completed(futures):
                i, j, r5, r20, rmax, overfill, err = future.result()

                Rise_5s_grid[i, j] = r5
                Rise_20s_grid[i, j] = r20
                Rise_Max_grid[i, j] = rmax
                Overfill_grid[i, j] = overfill

                completed_count += 1
                if completed_count % 50 == 0:
                    print(f"   [{completed_count}/{total_sims}] Completed...")

        print(f"   [{total_sims}/{total_sims}] All simulations done.")

        # --- Export to Excel ---
        df_5s = pd.DataFrame(Rise_5s_grid, index=h_range, columns=d_range)
        df_20s = pd.DataFrame(Rise_20s_grid, index=h_range, columns=d_range)
        df_max = pd.DataFrame(Rise_Max_grid, index=h_range, columns=d_range)
        df_over = pd.DataFrame(Overfill_grid, index=h_range, columns=d_range)

        filename = f"results_flow_{vent_rate}_lbm_s.xlsx"
        full_path = os.path.join(OUTPUT_DIR, filename)

        print(f"   [*] Saving data to {filename}...")
        try:
            with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
                df_5s.to_excel(writer, sheet_name='Rise_5s_in')
                df_20s.to_excel(writer, sheet_name='Rise_20s_in')
                df_max.to_excel(writer, sheet_name='Rise_Max_in')
                df_over.to_excel(writer, sheet_name='Overfill_Flag')
            print(f"   [*] Saved: {full_path}")
        except Exception as e:
            print(f"   [!] Error saving Excel file: {e}")

        # --- Generate Plots ---
        print("   [*] Generating plots...")
        plot_geometry_sweep(D_grid, H_grid, Rise_5s_grid, Rise_20s_grid,
                            Rise_Max_grid, Overfill_grid,
                            vent_rate, OUTPUT_DIR)

    end_time = time.time()
    print(f"\nAll Cases Completed. Total Time: {end_time - start_time:.2f} seconds")
