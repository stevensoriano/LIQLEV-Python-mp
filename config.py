# -*- coding: utf-8 -*-
"""
Configuration for LIQLEV-Python-parallel.
All user-tunable parameters are defined here.
"""
import os
import numpy as np
import pandas as pd
from thermo_utils import Tsat, Psat, DensitySat, sli

# =============================================================================
# USER INPUTS — Modify these as needed
# =============================================================================

# --- Parallel Execution ---
# Number of parallel worker processes.
# Set to an integer (e.g., 6, 12, 24).
# If set to None, auto-detects from SLURM_CPUS_PER_TASK (minus 1), or defaults to 6.
MAX_WORKERS = None

# --- Fluid ---
FLUID = "Nitrogen"

# --- Vent Rates to Sweep (lbm/s) ---
VENT_RATES = [0.003, 0.004, 0.005, 0.006]

# --- Fill Fraction ---
SWEEP_FILL = 0.25

# --- Geometry Sweep Ranges (feet) ---
D_RANGE = np.linspace(0.19685, 1.0, 50)    # Tank diameter sweep
H_RANGE = np.linspace(0.328084, 2.5, 40)   # Tank height sweep

# --- Pressure (psia) ---
PINIT_PSIA = 14.7
PFINAL_PSIA = 5.0

# --- Simulation Duration (seconds) ---
DURATION = 20.0

# --- Time Step (seconds) ---
DELTA_T = 0.01

# --- Gravity Settings ---
GRAVITY_FILE = os.path.join(os.path.dirname(__file__), 'data',
                            '5s_drop_tower_extracted_az_positive_data.csv')
HOLD_G_VALUE = 0.0014  # Gravity value (g's) to hold after CSV data ends

# --- Output Directory ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')

# --- Time Snapshots for Rise Extraction (seconds) ---
SNAPSHOT_5S = 5.0
SNAPSHOT_20S = 20.0

# --- Baseline Marker (smallest geometry in sweep) ---
BASELINE_D = 0.19685   # feet
BASELINE_H = 0.328084  # feet

# --- Target Rise Line (inches) ---
TARGET_RISE_IN = 2.0


# =============================================================================
# WORKER COUNT RESOLUTION
# =============================================================================
def get_max_workers():
    """Resolve the number of parallel workers."""
    if MAX_WORKERS is not None:
        return MAX_WORKERS
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        return max(1, int(slurm_cpus) - 1)
    return 6


# =============================================================================
# CONFIG & INPUT BUILDERS
# =============================================================================
def get_config(dtank_ft=None, tank_height_ft=None):
    """
    Build a configuration dictionary for a single geometry case.

    Parameters
    ----------
    dtank_ft : float, optional
        Tank diameter in feet. Defaults to BASELINE_D.
    tank_height_ft : float, optional
        Tank height in feet. Defaults to BASELINE_H.

    Returns
    -------
    dict
        Configuration dictionary with gravity profile and tank geometry.
    """
    config = {}
    config['FLUID'] = FLUID
    config['DURATION'] = DURATION

    if dtank_ft is None:
        dtank_ft = BASELINE_D
    if tank_height_ft is None:
        tank_height_ft = BASELINE_H

    config['DTANK'] = dtank_ft
    config['TANK_HEIGHT'] = tank_height_ft
    config['VOLT'] = (np.pi / 4) * (dtank_ft ** 2) * tank_height_ft
    config['GRAVITY_FUNCTION'] = None

    try:
        g_df = pd.read_csv(GRAVITY_FILE)
        g_to_ft_s2 = 32.174

        tggo_g = g_df['normalized_time'].to_numpy()
        xggo_g = g_df['az_positive'].to_numpy()

        last_original_data_time = tggo_g[-1]
        config['LAST_ORIGINAL_GRAVITY_TIME'] = last_original_data_time

        if DURATION > last_original_data_time:
            tggo_g = np.append(tggo_g, last_original_data_time + 1e-9)
            xggo_g = np.append(xggo_g, HOLD_G_VALUE)
            tggo_g = np.append(tggo_g, DURATION)
            xggo_g = np.append(xggo_g, HOLD_G_VALUE)

        config['TGGO_g'] = tggo_g
        config['XGGO_g'] = xggo_g
        config['NGGO'] = len(tggo_g)
        config['TGGO'] = tggo_g
        config['XGGO'] = xggo_g * g_to_ft_s2

    except FileNotFoundError:
        config['NGGO'] = 2
        config['TGGO'] = np.array([0.0, DURATION])
        config['XGGO'] = np.array([0.00032174, 0.00032174])
        config['TGGO_g'] = config['TGGO']
        config['XGGO_g'] = config['XGGO'] / 32.174
        config['LAST_ORIGINAL_GRAVITY_TIME'] = DURATION

    return config


def get_base_inputs(config, vent_rate_lbm_s, fill_fraction, neps=None):
    """
    Convert user-friendly config into the rigid solver input dictionary.

    Parameters
    ----------
    config : dict
        From get_config().
    vent_rate_lbm_s : float
        Vent mass flow rate (lbm/s).
    fill_fraction : float
        Initial liquid fill fraction (0-1).
    neps : int or None
        Epsilon schedule point count. None = varying, 0 = height_dep.

    Returns
    -------
    dict
        Input dictionary for liqlev_simulation().
    """
    fluid = config['FLUID']
    duration = config['DURATION']
    dtank = config['DTANK']
    volt = config['VOLT']
    NGGO = config['NGGO']
    TGGO = config['TGGO']
    XGGO = config['XGGO']

    PSI_TO_KPA = 6.89475729
    press_kpa = PINIT_PSIA * PSI_TO_KPA
    tinit = Tsat(fluid, press_kpa) * 1.8  # Rankine

    if neps is None:
        neps = 11
        teps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                         6.0, 7.0, 8.0, 9.0, duration])
        xeps = np.array([0.0000, 0.0513, 0.1780, 0.2800, 0.3620,
                         0.4220, 0.4700, 0.5200, 0.5600, 0.6000, 0.6000])
    else:
        teps = np.array([0.0, duration])
        xeps = np.array([0.0, 0.0])

    ac = 0.7854 * (dtank ** 2)
    htank = volt / ac
    Htzero = fill_fraction * htank

    if fluid == "Hydrogen":
        rhol = (0.1709 + 0.7454 * tinit - 0.04421 * tinit**2
                + 0.001248 * tinit**3 - 1.738e-5 * tinit**4
                + 9.424e-8 * tinit**5)
    else:
        t_k = tinit / 1.8
        ps_kpa = Psat(fluid, t_k)
        rhol = DensitySat(fluid, "liquid", ps_kpa) * 0.0624279606

    xmlzro = rhol * (Htzero * ac)

    tvmdot_arr = np.array([0.0, duration])
    xvmdot_arr = np.array([vent_rate_lbm_s, vent_rate_lbm_s])
    nvmd_count = len(tvmdot_arr)

    inputs = {
        "Title": "2D Sweep",
        "Liquid": fluid,
        "Units": "British",
        "Delta": DELTA_T,
        "Dtank": dtank,
        "Htzero": Htzero,
        "Volt": volt,
        "Xmlzro": xmlzro,
        "Pinit": PINIT_PSIA,
        "Pfinal": PFINAL_PSIA,
        "Tinit": tinit,
        "Thetin": 0.0,
        "Nvmd": nvmd_count,
        "Neps": neps,
        "Nlattm": 2,
        "Nvertm": 2,
        "Nggo": NGGO,
        "Tvmdot": tvmdot_arr,
        "Xvmdot": xvmdot_arr,
        "Teps": teps,
        "Xeps": xeps,
        "Tspal": np.array([0.0, duration]),
        "Xspacl": np.array([1.0, 1.0]),
        "Tspav": np.array([0.0, duration]),
        "Xspacv": np.array([1.0, 1.0]),
        "Tggo": TGGO,
        "Xggo": XGGO,
        "gravity_function": config['GRAVITY_FUNCTION'],
    }
    return inputs
