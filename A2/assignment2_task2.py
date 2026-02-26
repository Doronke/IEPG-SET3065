"""
Assignment 2 – Task 2: Probabilistic OPF with Wind Location & Reactive Power
=============================================================================

This script extends the baseline PPF from prob_opf_ieee9_wind.py by:
  1. Varying the wind plant location: Default (Bus 8), Bus 5, Bus 7
  2. Varying the reactive power control mode of the wind plant:
       - Unit PF  (Q = 0)
       - 0.95 pf Overexcited  (Q > 0 → wind injects reactive power, raises V)
       - 0.95 pf Underexcited (Q < 0 → wind absorbs reactive power, lowers V)
  3. Comparing voltage magnitude variability and branch loading vs the baseline
     (Default location + Unit PF = same as prob_opf_ieee9_wind.py).

Methodology (Monte Carlo PPF, N = 100 samples):
  - Wind power follows a Weibull-distributed speed through a piecewise power curve.
  - Load P is drawn from a normal distribution around the nominal values; Q is
    kept at the original PQ ratio.
  - For each sample an economic OPF (runopp) is run; non-convergent samples are
    skipped (as noted in the assignment).
  - Results are collected per scenario and analysed statistically.

All random inputs are pre-generated once so every scenario sees identical samples
→ fair comparison between scenarios.

Figures are saved to ./A2/figures_task2/.
Helper functions from assignment2_wind_scenarios.py are reused where applicable.
"""

import copy
import os
import re
import random
import warnings

import pandapower as pp
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive: save to file, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

# =============================================================================
# Configuration
# =============================================================================

# ── output directory ──────────────────────────────────────────────────────────
FIG_DIR = './A2/figures_task2/'
os.makedirs(FIG_DIR, exist_ok=True)

# Reuse filename helpers from assignment2_wind_scenarios.py
slug    = lambda s: re.sub(r'[^\w]+', '_', s).strip('_')
savefig = lambda name: plt.savefig(os.path.join(FIG_DIR, name), dpi=150,
                                   bbox_inches='tight')

# ── load reference network ────────────────────────────────────────────────────
BASE_NET       = pp.from_excel('./A2/ieee9-wind.xlsx')
WIND_TRAFO_IDX = 3     # trafo row that couples the wind subsystem to the 230 kV ring

N_BUSES  = len(BASE_NET.bus)
N_LINES  = len(BASE_NET.line)
N_TRAFOS = len(BASE_NET.trafo)
N_LOADS  = len(BASE_NET.load)

# =============================================================================
# Wind power curve parameters  (same as prob_opf_ieee9_wind.py → baseline match)
# =============================================================================
K       = 2.02   # Weibull shape parameter  [–]
LAMBDA_ = 11     # Weibull scale parameter  [m/s]
P_WPP   = 180    # Rated wind plant power   [MW]
W_IN    = 3      # Cut-in  wind speed       [m/s]
W_R     = 12     # Rated   wind speed       [m/s]
W_OUT   = 20     # Cut-off wind speed       [m/s]

def wind_power(ws: float) -> float:
    """Piecewise power curve: wind speed → active power [MW]."""
    if ws < W_IN or ws >= W_OUT:
        return 0.0
    elif ws < W_R:
        # Linear interpolation in the partial-load region (cubic in speed)
        return P_WPP * (ws**3 - W_IN**3) / (W_R**3 - W_IN**3)
    else:
        return P_WPP   # rated output in full-load region

# =============================================================================
# Pre-generate ALL random inputs once  (same seed as prob_opf_ieee9_wind.py)
# This ensures every scenario sees the exact same N operating conditions,
# making the comparison between scenarios fair.
# =============================================================================
N = 100
np.random.seed(5489)   # identical seed to the baseline file

# Wind power samples – Python's random module (Weibull), matches baseline
Pwpp_act = np.array([wind_power(random.weibullvariate(LAMBDA_, K))
                     for _ in range(N)])

# Load standard deviations drawn once from numpy random (matches baseline)
load_means   = np.asarray(BASE_NET.load.p_mw)
load_sd      = np.random.uniform(1, 30, N_LOADS)
pq_ratio     = np.asarray(BASE_NET.load.q_mvar / BASE_NET.load.p_mw)

# Pre-generate N load demand samples (N × N_LOADS matrices)
# Each row i is one set of random load values for sample i.
load_p_mat = np.vstack([np.random.normal(load_means, load_sd)
                        for _ in range(N)])           # shape (N, N_LOADS)
load_q_mat = load_p_mat * pq_ratio                   # fixed PQ ratio per load

# =============================================================================
# Scenario definitions
# =============================================================================

# Wind locations: pandapower 0-indexed bus numbers.
# 'None' keeps the HV bus from the Excel file (= Bus 8, the baseline).
# Bus 5 and Bus 7 are the alternative injection points tested in Task 2.
WIND_LOCATIONS = {
    "Default (Bus 8)": None,   # same as prob_opf_ieee9_wind.py → BASELINE
    "Bus 5":           5,
    "Bus 7":           7,
}

# Reactive power modes for the wind plant.
# Power factor 0.95 → angle φ = arccos(0.95) → tan(φ) ≈ 0.3287
# Overexcited  (+Q): wind injects reactive power → tends to raise bus voltages
# Underexcited (–Q): wind absorbs reactive power → tends to lower bus voltages
PF_TARGET = 0.95
Q_TAN     = np.tan(np.arccos(PF_TARGET))   # ≈ 0.3287

PF_MODES = {
    "Unit PF (Q=0)":          0.0,    # reference mode (baseline)
    "0.95 pf Overexcited":   +Q_TAN,  # Q = +P·tan(φ)  → voltage support
    "0.95 pf Underexcited":  -Q_TAN,  # Q = −P·tan(φ)  → voltage depression
}

# =============================================================================
# Helper: build network for a given wind location
# =============================================================================
def build_net(hv_bus):
    """
    Deep-copy BASE_NET and optionally redirect the wind-coupling transformer
    (WIND_TRAFO_IDX) to the specified HV bus.  hv_bus=None keeps the default.
    """
    net = copy.deepcopy(BASE_NET)
    if hv_bus is not None:
        net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = hv_bus
    return net

# =============================================================================
# Core PPF loop for one scenario
# =============================================================================
def run_ppf_scenario(hv_bus, q_multiplier, label=""):
    """
    Run a probabilistic OPF (Monte Carlo, N samples) for one scenario.

    Parameters
    ----------
    hv_bus       : int or None – pandapower bus index for wind coupling trafo
    q_multiplier : float – reactive power factor: q_wind = P_wind × q_multiplier
    label        : str  – display name for progress output

    Returns
    -------
    dict with:
      vm_pu      : ndarray (n_conv, N_BUSES)  – bus voltages per converged sample
      line_load  : ndarray (n_conv, N_LINES)  – line loadings [%]
      trafo_load : ndarray (n_conv, N_TRAFOS) – trafo loadings [%]
      n_conv     : int – number of converged OPF samples
    """
    # Build the topology for this scenario (done once per scenario)
    net_template = build_net(hv_bus)

    vm_list, ll_list, tl_list = [], [], []
    n_conv = 0

    for i, p in enumerate(Pwpp_act):
        # Fresh copy of the template for each sample (avoids state carry-over)
        net = copy.deepcopy(net_template)

        # Apply pre-generated random load values for sample i
        net.load.p_mw   = load_p_mat[i]
        net.load.q_mvar = load_q_mat[i]

        # Set wind active power and FIXED reactive power for this sample.
        # Setting max/min_q_mvar equal to each other forces the OPF to use
        # exactly q_wind without optimising Q (task requirement).
        q_wind = p * q_multiplier
        net.sgen['p_mw']       = p
        net.sgen['max_p_mw']   = p          # fixes P dispatch to actual wind
        net.sgen['min_p_mw']   = p
        net.sgen['q_mvar']     = q_wind
        net.sgen['max_q_mvar'] = q_wind     # fix Q → OPF cannot adjust it
        net.sgen['min_q_mvar'] = q_wind

        try:
            # Warm-start OPF with a plain PF solution (improves convergence rate)
            pp.runpp(net, verbose=False)
            pp.runopp(net, init='pf', verbose=False)

            # Store converged results
            vm_list.append(net.res_bus.vm_pu.values.copy())
            ll_list.append(net.res_line.loading_percent.values.copy())
            tl_list.append(net.res_trafo.loading_percent.values.copy())
            n_conv += 1

        except Exception:
            # Non-convergent sample: skip as per the assignment note.
            continue

    print(f"  {label:45s}: {n_conv}/{N} converged  "
          f"({100 * n_conv / N:.0f} %)")

    return {
        'vm_pu':      np.array(vm_list)  if vm_list  else np.empty((0, N_BUSES)),
        'line_load':  np.array(ll_list)  if ll_list  else np.empty((0, N_LINES)),
        'trafo_load': np.array(tl_list)  if tl_list  else np.empty((0, N_TRAFOS)),
        'n_conv':     n_conv,
    }

# =============================================================================
# Run all 9 scenarios  (3 locations × 3 PF modes)
# =============================================================================
print("Running Probabilistic OPF – Task 2  (N = 100 Monte Carlo samples)")
print("=" * 65)

all_results = {}
for loc_label, hv_bus in WIND_LOCATIONS.items():
    print(f"\n  Wind location: {loc_label}")
    for pf_label, q_mult in PF_MODES.items():
        key = (loc_label, pf_label)
        all_results[key] = run_ppf_scenario(
            hv_bus, q_mult, label=f"{loc_label} | {pf_label}")

# =============================================================================
# Convergence summary table
# =============================================================================
print("\n" + "=" * 65)
rows = [
    {'Wind Location': k[0], 'PF Mode': k[1],
     'Converged': v['n_conv'], 'Total': N,
     'Rate (%)': f"{100 * v['n_conv'] / N:.0f}"}
    for k, v in all_results.items()
]
print("\nCONVERGENCE SUMMARY")
print(pd.DataFrame(rows).to_string(index=False))
print()

# =============================================================================
# Plotting utilities
# =============================================================================
COLORS  = plt.rcParams['axes.prop_cycle'].by_key()['color']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']

# ── Trafo tick labels with connectivity info ──────────────────────────────────
trafo_ticks = [
    f"T{i} [WIND]" if i == WIND_TRAFO_IDX else f"T{i}"
    for i in BASE_NET.trafo.index
]

def errorbar_plot(ax, data_dict, idx, data_key, title, ylabel,
                  hline=None, hline_color='red', hline_label=None,
                  xticklabels=None):
    """
    Plot mean ± std for each scenario.  Gives a quick visual read on how the
    average level and spread change across scenarios.

    Parameters
    ----------
    data_dict    : {label: result_dict}
    idx          : element indices (buses / lines / trafos)
    data_key     : 'vm_pu', 'line_load', or 'trafo_load'
    """
    for j, (label, res) in enumerate(data_dict.items()):
        arr = res[data_key]
        if arr.size == 0:
            ax.plot([], [], color=COLORS[j % len(COLORS)],
                    label=f"{label} (no data)")
            continue
        mean_v = arr.mean(axis=0)
        std_v  = arr.std(axis=0)
        ax.errorbar(idx, mean_v, yerr=std_v,
                    fmt=MARKERS[j % len(MARKERS)] + '-',
                    color=COLORS[j % len(COLORS)],
                    capsize=3, linewidth=1.5, markersize=5,
                    label=f"{label}  (n={res['n_conv']})")
    if hline is not None:
        ax.axhline(hline, color=hline_color, linestyle='--',
                   linewidth=1.2, label=hline_label, zorder=0)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(idx)
        ax.set_xticklabels(xticklabels, fontsize=8)
    else:
        ax.set_xticks(idx)


def boxplot_comparison(ax, data_dict, idx, data_key, title, ylabel,
                       hline=None, hline_label=None, xticklabels=None):
    """
    Draw side-by-side box plots for each scenario.  Box plots capture the full
    probability distribution (median, IQR, whiskers) – better for variability
    analysis than a simple mean ± std representation.

    Parameters
    ----------
    data_dict : {label: result_dict}
    idx       : element indices
    data_key  : 'vm_pu', 'line_load', or 'trafo_load'
    """
    n_el   = len(idx)
    n_scen = len(data_dict)
    box_w  = 0.6 / max(n_scen, 1)
    offsets = np.linspace(-(n_scen - 1) * box_w / 2,
                           (n_scen - 1) * box_w / 2, n_scen)

    for j, (label, res) in enumerate(data_dict.items()):
        arr   = res[data_key]
        color = COLORS[j % len(COLORS)]
        if arr.size == 0:
            ax.plot([], [], color=color, label=f"{label} (no data)")
            continue

        # Each element gets one box; data_dict ordering determines colour/offset
        bp_data   = [arr[:, i] for i in range(n_el)]
        positions = np.arange(n_el) + offsets[j]
        ax.boxplot(
            bp_data, positions=positions, widths=box_w * 0.85,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.5),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        # Proxy artist for the legend
        ax.plot([], [], color=color, linewidth=5, alpha=0.5,
                label=f"{label}  (n={res['n_conv']})")

    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--',
                   linewidth=1.2, label=hline_label, zorder=0)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(np.arange(n_el))
        ax.set_xticklabels(xticklabels, fontsize=8)
    else:
        ax.set_xticks(np.arange(n_el))

# =============================================================================
# Figure set A: per-location comparison of PF modes
# (3 figures, one per wind location – shows effect of reactive power mode)
# For each location compare Unit PF vs Overexcited vs Underexcited.
# =============================================================================
bus_idx   = np.arange(N_BUSES)
line_idx  = np.arange(N_LINES)
trafo_idx = np.arange(N_TRAFOS)
bus_ticks   = [f"B{i}" for i in range(N_BUSES)]
line_ticks  = [f"L{i}" for i in range(N_LINES)]

for loc_label in WIND_LOCATIONS:
    subset = {pf: all_results[(loc_label, pf)] for pf in PF_MODES}

    # ── A1: Voltage magnitude – mean ± std and box plots ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Left panel: mean ± std (quick comparison of level and spread)
    errorbar_plot(
        axes[0], subset, bus_idx, 'vm_pu',
        f"Voltage Mean ± Std  |  Wind @ {loc_label}",
        "Voltage (pu)",
        hline=1.05, hline_label="V$_{max}$ = 1.05 pu",
        xticklabels=bus_ticks,
    )
    axes[0].axhline(0.95, color='blue', linestyle='--', linewidth=1.2,
                    label='V$_{min}$ = 0.95 pu')
    axes[0].set_ylim(0.88, 1.12)
    axes[0].legend(fontsize=7, loc='lower right')

    # Right panel: box plots (full distribution; reveals skew and outliers)
    boxplot_comparison(
        axes[1], subset, bus_idx, 'vm_pu',
        f"Voltage Distribution  |  Wind @ {loc_label}",
        "Voltage (pu)",
        hline=1.05, hline_label="V$_{max}$ = 1.05 pu",
        xticklabels=bus_ticks,
    )
    axes[1].axhline(0.95, color='blue', linestyle='--', linewidth=1.2,
                    label='V$_{min}$ = 0.95 pu')
    axes[1].set_ylim(0.88, 1.12)
    axes[1].legend(fontsize=7, loc='lower right')

    fig.suptitle(
        f"Voltage Variability – PF Mode Comparison  |  Wind @ {loc_label}  (N=100)",
        fontsize=12, fontweight='bold')
    savefig(f"A_voltage_pfmodes_{slug(loc_label)}.png")
    plt.close()

    # ── A2: Branch loading – lines + trafos ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), constrained_layout=True)

    errorbar_plot(
        axes[0], subset, line_idx, 'line_load',
        f"Line Loading Mean ± Std  |  Wind @ {loc_label}",
        "Loading (%)",
        hline=100, hline_label="100 % thermal limit",
        xticklabels=line_ticks,
    )
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7, loc='upper right')

    errorbar_plot(
        axes[1], subset, trafo_idx, 'trafo_load',
        f"Transformer Loading Mean ± Std  |  Wind @ {loc_label}",
        "Loading (%)",
        hline=100, hline_label="100 % thermal limit",
        xticklabels=trafo_ticks,
    )
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=7, loc='upper right')

    fig.suptitle(
        f"Branch Loading – PF Mode Comparison  |  Wind @ {loc_label}  (N=100)",
        fontsize=12, fontweight='bold')
    savefig(f"A_branch_pfmodes_{slug(loc_label)}.png")
    plt.close()

# =============================================================================
# Figure set B: per-PF-mode comparison of wind locations
# (3 figures, one per PF mode – shows effect of wind location)
# For each PF mode compare Default vs Bus 5 vs Bus 7.
# =============================================================================
for pf_label in PF_MODES:
    subset = {loc: all_results[(loc, pf_label)] for loc in WIND_LOCATIONS}

    # ── B1: Voltage ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    errorbar_plot(
        axes[0], subset, bus_idx, 'vm_pu',
        f"Voltage Mean ± Std  |  {pf_label}",
        "Voltage (pu)",
        hline=1.05, hline_label="V$_{max}$ = 1.05 pu",
        xticklabels=bus_ticks,
    )
    axes[0].axhline(0.95, color='blue', linestyle='--', linewidth=1.2,
                    label='V$_{min}$ = 0.95 pu')
    axes[0].set_ylim(0.88, 1.12)
    axes[0].legend(fontsize=7, loc='lower right')

    boxplot_comparison(
        axes[1], subset, bus_idx, 'vm_pu',
        f"Voltage Distribution  |  {pf_label}",
        "Voltage (pu)",
        hline=1.05, hline_label="V$_{max}$ = 1.05 pu",
        xticklabels=bus_ticks,
    )
    axes[1].axhline(0.95, color='blue', linestyle='--', linewidth=1.2,
                    label='V$_{min}$ = 0.95 pu')
    axes[1].set_ylim(0.88, 1.12)
    axes[1].legend(fontsize=7, loc='lower right')

    fig.suptitle(
        f"Voltage Variability – Location Comparison  |  {pf_label}  (N=100)",
        fontsize=12, fontweight='bold')
    savefig(f"B_voltage_location_{slug(pf_label)}.png")
    plt.close()

    # ── B2: Branch loading ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), constrained_layout=True)

    errorbar_plot(
        axes[0], subset, line_idx, 'line_load',
        f"Line Loading Mean ± Std  |  {pf_label}",
        "Loading (%)", hline=100, hline_label="100 % limit",
        xticklabels=line_ticks,
    )
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7, loc='upper right')

    errorbar_plot(
        axes[1], subset, trafo_idx, 'trafo_load',
        f"Transformer Loading Mean ± Std  |  {pf_label}",
        "Loading (%)", hline=100, hline_label="100 % limit",
        xticklabels=trafo_ticks,
    )
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=7, loc='upper right')

    fig.suptitle(
        f"Branch Loading – Location Comparison  |  {pf_label}  (N=100)",
        fontsize=12, fontweight='bold')
    savefig(f"B_branch_location_{slug(pf_label)}.png")
    plt.close()

# =============================================================================
# Figure C: Voltage Std Summary – scalar per-bus std averaged over buses
# Bar chart showing mean std(vm_pu) per scenario.
# A lower value means the voltage profile is more predictable (less uncertain).
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))
scenario_labels = [f"{l}\n{p}" for (l, p) in all_results.keys()]
mean_stds = [
    res['vm_pu'].std(axis=0).mean() if res['vm_pu'].size > 0 else 0.0
    for res in all_results.values()
]
bar_colors = [COLORS[i % len(COLORS)] for i in range(len(scenario_labels))]
bars = ax.bar(range(len(scenario_labels)), mean_stds,
              color=bar_colors, edgecolor='k', linewidth=0.5, alpha=0.8)

# Mark the baseline bar with a star annotation
baseline_key = ("Default (Bus 8)", "Unit PF (Q=0)")
baseline_idx  = list(all_results.keys()).index(baseline_key)
ax.annotate("Baseline", xy=(baseline_idx, mean_stds[baseline_idx]),
            xytext=(baseline_idx, mean_stds[baseline_idx] + 0.001),
            ha='center', fontsize=8, color='black',
            arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xticks(range(len(scenario_labels)))
ax.set_xticklabels(scenario_labels, fontsize=7)
ax.set_ylabel("Mean Voltage Std (pu)  [lower = more stable]")
ax.set_title("Voltage Variability Summary – All Scenarios  (N=100 PPF samples)",
             fontsize=11, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
savefig("C_voltage_std_summary.png")
plt.close()

# =============================================================================
# Figure D: Baseline vs all other scenarios – direct delta comparison
# Shows Δmean(V) and Δstd(V) vs the baseline for every bus.
# Baseline = Default (Bus 8) | Unit PF (Q=0)  (= prob_opf_ieee9_wind.py)
# =============================================================================
baseline_res = all_results[baseline_key]
baseline_mean = baseline_res['vm_pu'].mean(axis=0)
baseline_std  = baseline_res['vm_pu'].std(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
for j, (key, res) in enumerate(all_results.items()):
    if key == baseline_key:
        continue   # skip baseline itself
    if res['vm_pu'].size == 0:
        continue
    loc_label, pf_label = key
    label = f"{loc_label} | {pf_label}  (n={res['n_conv']})"
    delta_mean = res['vm_pu'].mean(axis=0) - baseline_mean
    delta_std  = res['vm_pu'].std(axis=0)  - baseline_std

    axes[0].plot(bus_idx, delta_mean, MARKERS[j % len(MARKERS)] + '-',
                 color=COLORS[j % len(COLORS)], linewidth=1.5, markersize=5,
                 label=label)
    axes[1].plot(bus_idx, delta_std, MARKERS[j % len(MARKERS)] + '-',
                 color=COLORS[j % len(COLORS)], linewidth=1.5, markersize=5,
                 label=label)

for ax, title, ylabel in [
    (axes[0], "Δ Mean Voltage vs Baseline (positive = higher voltage)", "Δ V_mean (pu)"),
    (axes[1], "Δ Voltage Std vs Baseline (positive = more variable)", "Δ V_std (pu)"),
]:
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', label='Baseline')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(bus_idx)
    ax.set_xticklabels(bus_ticks, fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

fig.suptitle("Deviation from Baseline – Voltage Mean and Variability  (N=100)",
             fontsize=12, fontweight='bold')
savefig("D_voltage_delta_vs_baseline.png")
plt.close()

# =============================================================================
# Figure E: Baseline vs all other scenarios – branch loading delta
# Δmean line/trafo loading vs baseline.
# =============================================================================
baseline_ll_mean = baseline_res['line_load'].mean(axis=0)
baseline_tl_mean = baseline_res['trafo_load'].mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
for j, (key, res) in enumerate(all_results.items()):
    if key == baseline_key:
        continue
    if res['line_load'].size == 0:
        continue
    loc_label, pf_label = key
    label = f"{loc_label} | {pf_label}  (n={res['n_conv']})"
    delta_ll = res['line_load'].mean(axis=0)  - baseline_ll_mean
    delta_tl = res['trafo_load'].mean(axis=0) - baseline_tl_mean
    axes[0].plot(line_idx,  delta_ll, MARKERS[j % len(MARKERS)] + '-',
                 color=COLORS[j % len(COLORS)], linewidth=1.5, markersize=5,
                 label=label)
    axes[1].plot(trafo_idx, delta_tl, MARKERS[j % len(MARKERS)] + '-',
                 color=COLORS[j % len(COLORS)], linewidth=1.5, markersize=5,
                 label=label)

for ax, idx_arr, ticks, title, ylabel in [
    (axes[0], line_idx,  line_ticks,  "Δ Mean Line  Loading vs Baseline", "Δ Loading (%)"),
    (axes[1], trafo_idx, trafo_ticks, "Δ Mean Trafo Loading vs Baseline", "Δ Loading (%)"),
]:
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', label='Baseline')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(idx_arr)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

fig.suptitle("Deviation from Baseline – Branch Loading  (N=100)",
             fontsize=12, fontweight='bold')
savefig("E_branch_delta_vs_baseline.png")
plt.close()

print(f"\nAll figures saved to  {os.path.abspath(FIG_DIR)}")
print("\nFigures produced:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
