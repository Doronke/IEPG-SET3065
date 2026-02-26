"""
Assignment 2 – Economic Dispatch: Wind Power Plant Location & Demand Scaling
=============================================================================
Scenarios investigated
  • Wind connected at bus 4 (baseline), bus 5, or bus 7  (pandapower 0-indexed)
  • Demand scaling: 100 %, 150 %, 200 %

One-line diagram (pandapower indices)
  Slack (bus 0, 16.5 kV)  ──T0── bus 3
  Gen2  (bus 1, 18   kV)  ──T2── bus 7
  Gen3  (bus 2, 13.8 kV)  ──T1── bus 5
  Transmission ring: 3-4-5-6-7-8-3  (230 kV)
  Loads: bus 4 (90 MW), bus 6 (100 MW), bus 8 (125 MW)
  Wind subsystem: bus 9 (110 kV) ──line── bus 10 ──T4── bus 11 (33 kV, sgen)
  Wind coupling transformer T3:  hv_bus = {4/5/7}  lv_bus = 9

Both active (P) and reactive (Q) power are considered:
  • Load P and Q are both scaled by the demand factor.
  • Generator Q limits (min/max_q_mvar) are read from the Excel file and enforced
    by the OPF.  Cost coefficients are on P only (standard economic dispatch).
  • Wind (sgen) operates at unity power factor (max/min_q_mvar = 0 in Excel).

For OPF-infeasible cases a plain power flow (runpp) is executed as a fallback so
that voltages and branch loadings are still available for analysis.  These results
are clearly marked in every plot (hatched bars, "OPF Infeas." labels).
"""

import copy, os, re, warnings
import pandapower as pp
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive: save only, no pop-up windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
warnings.filterwarnings('ignore', message='Tight layout not applied.*')
warnings.filterwarnings('ignore', message='constrained_layout not applied.*')

pd.options.display.max_columns = None

# ── output directory ──────────────────────────────────────────────────────────
FIG_DIR = './A2/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
slug  = lambda s: re.sub(r'[^\w]+', '_', s).strip('_')   # safe filename helper
savefig = lambda name: plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches='tight')

# ── load reference network ────────────────────────────────────────────────────
BASE_NET       = pp.from_excel('./A2/ieee9-wind.xlsx')
WIND_TRAFO_IDX = 3      # trafo row coupling wind subsystem to the 9-bus ring

N_BUSES  = len(BASE_NET.bus)
N_LINES  = len(BASE_NET.line)
N_TRAFOS = len(BASE_NET.trafo)

# ── scenario definitions ──────────────────────────────────────────────────────
WIND_LOCATIONS = {"Bus 8 (Baseline)": 8, "Bus 5": 5, "Bus 7": 7}
DEMAND_SCALES  = {"100pct": 1.0, "150pct": 1.5, "200pct": 2.0}

# =============================================================================
# Run all 9 scenarios
# =============================================================================
results = {}

for loc_label, hv_bus in WIND_LOCATIONS.items():
    for dem_label, scale in DEMAND_SCALES.items():
        net = copy.deepcopy(BASE_NET)
        net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = hv_bus
        # Scale both active AND reactive demand
        net.load['p_mw']   = BASE_NET.load['p_mw']   * scale
        net.load['q_mvar'] = BASE_NET.load['q_mvar']  * scale

        key     = (loc_label, dem_label)
        opf_ok  = False

        for init in ['pf', 'dc', 'flat']:           # try OPF with fallback inits
            try:
                n = copy.deepcopy(net)
                pp.runopp(n, init=init, verbose=False)
                results[key] = dict(opf_ok=True, cost=n.res_cost,
                                    vm_pu=n.res_bus['vm_pu'].values.copy(),
                                    line_load=n.res_line['loading_percent'].values.copy(),
                                    trafo_load=n.res_trafo['loading_percent'].values.copy(),
                                    p_gen_ext=n.res_ext_grid['p_mw'].values[0],
                                    q_gen_ext=n.res_ext_grid['q_mvar'].values[0],
                                    p_gen=n.res_gen['p_mw'].values.copy(),
                                    q_gen=n.res_gen['q_mvar'].values.copy())
                opf_ok = True
                print(f"[OPF OK ] {loc_label:22s} | {dem_label.replace('pct',' %')} | cost={n.res_cost:,.1f} EUR  (init={init})")
                break
            except Exception:
                pass

        if not opf_ok:                              # plain-PF fallback
            try:
                n = copy.deepcopy(net)
                n.gen['p_mw']  = n.gen['max_p_mw']     # dispatch at rated P
                n.sgen['p_mw'] = n.sgen['max_p_mw']
                pp.runpp(n, verbose=False)
                results[key] = dict(opf_ok=False, cost=None,
                                    vm_pu=n.res_bus['vm_pu'].values.copy(),
                                    line_load=n.res_line['loading_percent'].values.copy(),
                                    trafo_load=n.res_trafo['loading_percent'].values.copy(),
                                    p_gen_ext=n.res_ext_grid['p_mw'].values[0],
                                    q_gen_ext=n.res_ext_grid['q_mvar'].values[0],
                                    p_gen=n.res_gen['p_mw'].values.copy(),
                                    q_gen=n.res_gen['q_mvar'].values.copy())
                print(f"[PF FALL] {loc_label:22s} | {dem_label.replace('pct',' %')} | OPF infeasible – PF fallback")
            except Exception as e:
                results[key] = dict(opf_ok=False, cost=None, vm_pu=None,
                                    line_load=None, trafo_load=None,
                                    p_gen_ext=None, q_gen_ext=None,
                                    p_gen=None, q_gen=None)
                print(f"[FAILED ] {loc_label:22s} | {dem_label.replace('pct',' %')} | {e}")

print("\n" + "=" * 70)

# =============================================================================
# Summary table
# =============================================================================
rows = [{'Wind Location': loc, 'Demand': dem.replace('pct', ' %'),
         'OPF': 'OK' if r['opf_ok'] else 'INFEASIBLE',
         'Cost (EUR)': f"{r['cost']:,.1f}" if r['cost'] is not None else "—",
         'Ext P (MW)': f"{r['p_gen_ext']:.1f}" if r['p_gen_ext'] is not None else "—",
         'Ext Q (MVAr)': f"{r['q_gen_ext']:.1f}" if r['q_gen_ext'] is not None else "—"}
        for (loc, dem), r in results.items()]
print("\nSUMMARY TABLE")
print(pd.DataFrame(rows).to_string(index=False))
print()

# =============================================================================
# Plotting
# =============================================================================
COLORS  = plt.rcParams['axes.prop_cycle'].by_key()['color']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']
N_DEM   = len(DEMAND_SCALES)
N_LOC   = len(WIND_LOCATIONS)
W       = 0.22 / max(N_DEM, N_LOC)     # bar width

bus_idx   = np.arange(N_BUSES)
line_idx  = np.arange(N_LINES)
trafo_idx = np.arange(N_TRAFOS)

# trafo tick labels (mark wind coupling transformer)
trafo_ticks = [f"T{i} [WIND]\n(→{int(r.lv_bus)})" if i == WIND_TRAFO_IDX
               else f"T{i}\n({int(r.hv_bus)}→{int(r.lv_bus)})"
               for i, r in BASE_NET.trafo.iterrows()]

def _draw_bar(ax, idx, vals, offset, color, label, opf_ok):
    """Grouped bar: hatched + pale when OPF infeasible."""
    kw = dict(width=W, color=color, edgecolor='k', linewidth=0.4, label=label)
    if opf_ok:
        ax.bar(idx + offset, vals, alpha=0.85, **kw)
    else:
        ax.bar(idx + offset, vals, alpha=0.35, hatch='//', linewidth=0.5, **kw)

def _annotate_infeasible(ax, xpos):
    """Small 'OPF Infeas.' label at the bottom of each affected bar group."""
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * 0.02
    for xp in xpos:
        ax.text(xp, y, "OPF\nInfeas.", ha='center', va='bottom',
                fontsize=6.5, color='darkred', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))

def grouped_bar_figure(ax, series_iter, idx, data_key, n_series,
                       hline=None, hline_label=None):
    """
    Draw a grouped-bar chart on *ax*.

    series_iter : iterable of (label, result_dict, colour)
    idx         : np.arange of element positions
    data_key    : key into result_dict ('vm_pu', 'line_load', 'trafo_load')
    n_series    : total number of series (for offset calculation)
    """
    offsets     = np.linspace(-W*(n_series-1)/2, W*(n_series-1)/2, n_series)
    infeas_xpos = []
    for j, (label, res, color) in enumerate(series_iter):
        vals = res.get(data_key)
        if vals is None:
            # No bars drawn – just add a legend-only marker
            ax.plot([], [], 'x', color=color, markersize=8, markeredgewidth=2,
                    label=f"{label}  [FAILED]")
            continue
        lbl = f"{label}  [PF fallback]" if not res['opf_ok'] else label
        _draw_bar(ax, idx, vals, offsets[j], color, lbl, res['opf_ok'])
        if not res['opf_ok']:
            infeas_xpos.extend((idx + offsets[j]).tolist())
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=0.9, label=hline_label)
    _annotate_infeasible(ax, infeas_xpos)

# ── 5a. Voltage – one figure per demand level (all 3 wind locations) ──────────
for dem_label in DEMAND_SCALES:
    fig, ax = plt.subplots(figsize=(13, 5))
    series = [(l, results[(l, dem_label)], COLORS[j])
              for j, l in enumerate(WIND_LOCATIONS)]
    grouped_bar_figure(ax, series, bus_idx, 'vm_pu', N_LOC,
                       hline=1.05, hline_label='V$_{max}$ = 1.05 pu')
    ax.axhline(0.95, color='blue', linestyle='--', linewidth=0.9, label='V$_{min}$ = 0.95 pu')
    ax.set(xticks=bus_idx, xticklabels=[f"B{i}" for i in range(N_BUSES)],
           xlabel="Bus Index",  ylabel="Voltage (pu)",
           title=f"Voltage by Wind Location — Demand: {dem_label.replace('pct',' %')}")
    ax.set_ylim(0.85, 1.12)
    ax.legend(loc='lower right', fontsize=8)
    savefig(f"voltage_compare_{slug(dem_label)}.png")
    plt.close()

# ── 5b. Branch loading – one figure per demand level (all 3 wind locations) ───
for dem_label in DEMAND_SCALES:
    fig, axes = plt.subplots(2, 1, figsize=(13, 12), constrained_layout=True)
    series = [(l, results[(l, dem_label)], COLORS[j])
              for j, l in enumerate(WIND_LOCATIONS)]
    for ax, data_key, idx, ticks, xlabel, title in [
        (axes[0], 'line_load',  line_idx,  [f"L{i}" for i in range(N_LINES)],
         "Line Index",        f"Line Loading — Demand: {dem_label.replace('pct',' %')}"),
        (axes[1], 'trafo_load', trafo_idx, trafo_ticks,
         "Transformer Index", f"Transformer Loading — Demand: {dem_label.replace('pct',' %')}"),
    ]:
        grouped_bar_figure(ax, series, idx, data_key, N_LOC,
                           hline=100, hline_label='100 % limit')
        ax.set(xticks=idx, xlabel=xlabel, ylabel="Loading (%)", title=title)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
    savefig(f"branch_compare_{slug(dem_label)}.png")
    plt.close()

# ── 5c. Voltage comparison – +50 % and +100 % demand, side-by-side ───────────
elevated_demands = ["150pct", "200pct"]
fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True, constrained_layout=True)
for ax, dem_label in zip(axes, elevated_demands):
    series = [(l, results[(l, dem_label)], COLORS[j])
              for j, l in enumerate(WIND_LOCATIONS)]
    grouped_bar_figure(ax, series, bus_idx, 'vm_pu', N_LOC,
                       hline=1.05, hline_label='V$_{max}$ = 1.05 pu')
    ax.axhline(0.95, color='blue', linestyle='--', linewidth=0.9, label='V$_{min}$ = 0.95 pu')
    ax.set(xticks=bus_idx, xticklabels=[f"B{i}" for i in range(N_BUSES)],
           xlabel="Bus Index", ylabel="Voltage (pu)",
           title=f"Voltage by Wind Location — Demand: {dem_label.replace('pct',' %')}")
    ax.set_ylim(0.85, 1.12)
    ax.legend(loc='lower right', fontsize=8)
fig.suptitle("Voltage Comparison: All 3 Wind Locations under Elevated Demand",
             fontsize=12, fontweight='bold')
savefig("voltage_compare_elevated_demand.png")
plt.close()

# ── 5d. Dispatch cost vs demand level ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
dem_pct = [100, 150, 200]
for j, (loc_label, _) in enumerate(WIND_LOCATIONS.items()):
    costs = [results[(loc_label, d)]['cost'] for d in DEMAND_SCALES]
    ax.plot(dem_pct, [c if c is not None else np.nan for c in costs],
            marker=MARKERS[j], label=loc_label, color=COLORS[j], linewidth=2, markersize=8)
    for xp, cp in zip(dem_pct, costs):
        if cp is None:
            ax.plot(xp, 0, marker='x', markersize=14, color=COLORS[j], markeredgewidth=2)
            ax.text(xp, 200, "Infeasible", ha='center', va='bottom',
                    fontsize=8, color=COLORS[j], style='italic')

handles, labels_ = ax.get_legend_handles_labels()
infeas_patch = mpatches.Patch(facecolor='grey', alpha=0.3, hatch='//', edgecolor='k',
                               label='OPF Infeasible (PF fallback in other plots)')
ax.legend(handles=handles + [infeas_patch], fontsize=8)
ax.set(xlabel="Demand Level (%)", ylabel="Total Dispatch Cost (EUR)",
       title="Economic Dispatch Cost vs Demand Level", xticks=dem_pct)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
savefig("cost_vs_demand.png")
plt.close()

print(f"\nAll figures saved to  {os.path.abspath(FIG_DIR)}")
