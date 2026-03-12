# LIQLEV-Python-mp: Parallel Cryogenic Liquid Level Rise Geometry Sweep

A parallelized Python framework for running 2D geometry sweeps of the LIQLEV boundary-layer model. This tool predicts transient liquid level rise (LLR) across a grid of tank diameters and heights, enabling rapid parametric design studies for cryogenic propellant tanks under microgravity venting conditions.

> **Single-Node (Shared Memory) Only:** This code uses Python's `concurrent.futures.ProcessPoolExecutor` for parallelism. All worker processes run on a **single compute node** using shared memory. It is not designed for distributed multi-node execution. For SLURM clusters, use `--ntasks=1` with `--cpus-per-task=N` to allocate cores on one node.

## Overview

The management of cryogenic propellants under microgravity conditions presents significant challenges, particularly the safe venting of vapor without entraining liquid. When a cryogenic tank is vented to relieve internal pressure, the rapid pressure reduction initiates thermodynamic flashing and bulk boiling. Vapor generated within the superheated liquid bulk and along the wetted tank walls forms bubbles that displace the surrounding liquid, causing the liquid-vapor interface to rise. If this mixture reaches the vent port, liquid entrainment can occur, resulting in propellant loss.

This project builds on the serial [LIQLEV-Python](https://github.com/stevensoriano/LIQLEV-Python) implementation, which is itself a Python reimplementation of the original Fortran LIQLEV model developed following the Saturn S-IVB AS-203 flight experiment[^1] and subsequent adaptations for the Human Landing System (HLS) program[^2]. The parallel version adds:

- **Multiprocessing:** Geometry cases are distributed across configurable worker processes.
- **2D Geometry Sweep:** Sweeps over tank diameter and height for multiple vent flow rates.
- **Time Snapshots:** Extracts liquid level rise at 5s, 20s, and peak.
- **Contour Visualization:** Generates 3-panel contour plots per flow rate with overfill hatching.
- **Excel Export:** Saves computed rise grids to `.xlsx` files for post-processing.

## Technical Details & Mathematical Model

The LIQLEV model explicitly accounts for wall-driven boil-off and its contribution to interface rise by treating the sidewall as a surface on which a growing vapor film forms. Bubbles nucleate, grow, and detach within this film, displacing bulk liquid.

### Piston-Like Displacement

The framework retains the original one-dimensional piston-like displacement assumption, where all vapor generated is treated as uniformly distributed across the tank cross-section. The instantaneous interface location is tracked via the discrete relation:

$$Z_{\rm int}(t) = Z_{\rm int}(t-\Delta t) + \frac{\Delta V_{\rm BL} + \Delta m_{\rm liq}/\rho_{\rm liq}}{A_c}\Delta t$$

where $\Delta V_{\rm BL}$ is the incremental vapor volume generated in the wall thermal boundary layer, $\Delta m_{\rm liq}$ is the mass of liquid flashed to vapor, $\rho_{\rm liq}$ is the saturated liquid density, and $A_c$ is the tank cross-sectional area.

### Boil-Off Partitioning Ratio ($\epsilon$)

The parameter $\epsilon$ represents the instantaneous fraction of total vapor mass generated within the wall thermal boundary layer relative to the vapor generated at the liquid-vapor free surface. In the height-dependent mode, it is computed from the geometric wetted-area ratio:

$$\epsilon(t) = \frac{\pi D h(t)}{\pi D h(t) + \frac{\pi D^{2}}{4}}$$

### Thermodynamic Integration

Thermodynamic properties are provided by the open-source `CoolProp` library[^3], delivering consistent, high-accuracy real-fluid properties for Nitrogen, Hydrogen, Oxygen, and Methane across any pressure regime. The critical saturation-curve slope $(dP/dT)_{\rm sat}$ is computed dynamically via a central finite-difference scheme.

## Project Structure

```text
LIQLEV-Python-mp/
├── data/                 # Input gravity profiles (CSV)
├── results/              # Output directory (.xlsx and .png files)
├── config.py             # User configuration: workers, geometry, fluid, etc.
├── core.py               # LIQLEV transient solver
├── thermo_utils.py       # CoolProp thermodynamic wrappers
├── plotting.py           # Contour plot generation
├── main.py               # Parallel execution script
├── job.sbatch            # SLURM batch submission template
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/stevensoriano/LIQLEV-Python-mp.git
cd LIQLEV-Python-mp
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure the Simulation

Open `config.py` to set all simulation parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_WORKERS` | Number of parallel worker processes. Set to `None` for auto-detection. | `None` |
| `FLUID` | Cryogenic fluid (`"Nitrogen"`, `"Hydrogen"`, `"Oxygen"`, `"Methane"`) | `"Nitrogen"` |
| `VENT_RATES` | List of vent mass flow rates to sweep (lbm/s) | `[0.003, 0.004, 0.005, 0.006]` |
| `SWEEP_FILL` | Initial liquid fill fraction | `0.25` |
| `D_RANGE` | Diameter sweep range (ft), via `np.linspace` | `0.197 - 1.0`, 50 points |
| `H_RANGE` | Height sweep range (ft), via `np.linspace` | `0.328 - 2.5`, 40 points |
| `PINIT_PSIA` | Initial tank pressure (psia) | `14.7` |
| `PFINAL_PSIA` | Final target pressure (psia) | `5.0` |
| `DURATION` | Simulation duration (seconds) | `20.0` |

**Worker count resolution order:**
1. If `MAX_WORKERS` is set to an integer, that value is used.
2. If `MAX_WORKERS = None` and running under SLURM, uses `SLURM_CPUS_PER_TASK - 1`.
3. Otherwise, defaults to 6.

### 2. Manage Gravity Profiles

The simulation uses a transient gravity profile from a CSV file in the `data/` directory. The CSV must contain columns `normalized_time` (seconds) and `ax_positive` (g's). If the simulation duration exceeds the CSV data, the model holds the gravity at the configured `HOLD_G_VALUE` for the remainder.

If the gravity CSV is not found, the code falls back to a constant low-gravity default.

### 3. Run Locally

```bash
python main.py
```

### 4. Run on SLURM

Edit `job.sbatch` to set the desired CPU count and time limit, then submit:

```bash
sbatch job.sbatch
```

The job allocates a single node with the specified number of CPUs. The code automatically reads `SLURM_CPUS_PER_TASK` to set the worker count.

### 5. Outputs

For each vent rate in `VENT_RATES`, the script produces:

- **Excel file** (`results/results_flow_{rate}_lbm_s.xlsx`) with four sheets:
  - `Rise_5s_in` — Level rise at t=5s (inches), indexed by height (rows) and diameter (columns)
  - `Rise_20s_in` — Level rise at t=20s (inches)
  - `Rise_Max_in` — Peak level rise (inches)
  - `Overfill_Flag` — 1.0 if liquid reached the tank top, 0.0 otherwise

- **Contour plot** (`results/plots_flow_{rate}.png`) with three panels:
  - Rise at 5s, Rise at 20s, and Max Rise
  - Hatched regions indicate geometries where overfill occurred
  - Red contour line marks the 2.0" target rise threshold

## Parallelism Architecture

This code uses **single-node shared-memory parallelism** via Python's `concurrent.futures.ProcessPoolExecutor`:

- The 2D geometry grid (diameter x height) is flattened into a task list.
- Tasks are distributed across `N` worker processes on a single machine.
- Each worker independently constructs its configuration, runs the full LIQLEV solver, and returns scalar results (rise at 5s, 20s, max, and overfill flag).
- No shared state or inter-process communication is required.

This approach is well-suited for SLURM single-node allocations (`--ntasks=1 --cpus-per-task=N`) but **does not scale across multiple nodes**. For multi-node distributed execution, an MPI-based approach (e.g., `mpi4py`) would be required.

---

## References

[^1]: Bradshaw, R. D., "Evaluation and Application of Data from Low-Gravity Orbital Experiment, Phase I Final Report", General Dynamics, Convair Division, NASA CR-109847 (Report No. GDC-DDB-70-003), 1970.

[^2]: Moran, Matt, "Boundary Layer Model: Adaptation for HLS", Internal NASA Document, n.d.

[^3]: Bell, Ian H. et al., "Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp", Industrial & Engineering Chemistry Research, vol. 53, no. 6, pp. 2498-2508, 2014.
