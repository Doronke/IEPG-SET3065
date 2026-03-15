import copy
import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.max_columns = None

# -------------------------
# Load base network
# -------------------------
BASE_NET = pp.from_excel(
    r'C:\Users\dagma\Documents\Master\SET3065 Intelligent Electrical Power Grids\Assignments_v1-2\A2\ieee9-wind.xlsx'
)

# Transformer connecting wind plant
WIND_TRAFO_IDX = 3

# Wind locations to test (added Bus 8)
WIND_LOCATIONS = {
    "Bus 5": 5,
    "Bus 7": 7,
    "Bus 8": 8
}

# Load increase scenarios
LOAD_FACTORS = {
    "Base": 1.0,
    "50% Increase": 1.5,
    "100% Increase": 2.0
}

results = {}

# -------------------------
# Run OPF scenarios
# -------------------------
for label, bus in WIND_LOCATIONS.items():
    for factor_label, factor in LOAD_FACTORS.items():

        net = copy.deepcopy(BASE_NET)

        # Change wind connection
        net.trafo.at[WIND_TRAFO_IDX, "hv_bus"] = bus

        # Increase loads
        net.load["p_mw"] *= factor
        net.load["q_mvar"] *= factor

        # Run OPF
        try:
            pp.runopp(net, init='pf')
        except pp.optimal_powerflow.OPFNotConverged:
            print(f"OPF did not converge for {label} | {factor_label}")
            continue

        key = f"{label} | {factor_label}"
        results[key] = {
            "cost": net.res_cost,
            "voltages": net.res_bus.vm_pu.values,
            "line_loading": net.res_line.loading_percent.values,
            "trafo_loading": net.res_trafo.loading_percent.values
        }

        print(f"\n{key}")
        print(f"Total cost: {net.res_cost:.2f} EUR")

# -------------------------
# Helper function to plot grouped bar charts
# -------------------------
def plot_grouped_bar(data_dict, y_label, title, limit=None, element_names=None, element_key="voltages"):
    # Only keep scenarios that exist
    valid_keys = [k for k in data_dict if element_key in data_dict[k]]
    if not valid_keys:
        print(f"No valid data for {title}")
        return

    n_elements = len(data_dict[valid_keys[0]][element_key])
    n_scenarios = len(valid_keys)
    bar_width = 0.2
    x = np.arange(n_elements)

    plt.figure(figsize=(12,6))
    for i, key in enumerate(valid_keys):
        values = data_dict[key][element_key]
        plt.bar(x + i*bar_width, values, width=bar_width, label=key)

    if limit:
        plt.axhline(limit, color="r", linestyle="--", label="Limit")

    plt.xlabel("Element Index")
    plt.ylabel(y_label)
    plt.title(title)
    if element_names:
        plt.xticks(x + bar_width*(n_scenarios-1)/2, element_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# -------------------------
# Plot bus voltages per load scenario
# -------------------------
for factor_label in LOAD_FACTORS.keys():
    scenario_keys = [k for k in results if factor_label in k]
    if not scenario_keys:
        continue

    plt.figure(figsize=(6.5,3.5))
    x = np.arange(len(BASE_NET.bus))
    bar_width = 0.2

    for i, key in enumerate(scenario_keys):
        # Only the bus part (before '|')
        wind_bus = key.split('|')[0].strip()
        plt.bar(x + i*bar_width,
                results[key]["voltages"],
                width=bar_width,
                label=wind_bus)

    # Only one limit in legend
    plt.axhline(1.05, color='r', linestyle='--', label="Limit")  # upper limit
    plt.axhline(0.95, color='r', linestyle='--', label="_nolegend_")  # lower limit

    plt.ylim(0.9, 1.1)
    plt.xlabel("Bus Index")
    plt.ylabel("Voltage (pu)")
    plt.title(f"Bus Voltages - {factor_label}")  # Include demand increase
    plt.xticks(x + bar_width*(len(scenario_keys)-1)/2, [f"Bus {i}" for i in range(len(BASE_NET.bus))])
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()


# -------------------------
# Plot line loading per load scenario (compact)
# -------------------------
for factor_label in LOAD_FACTORS.keys():
    scenario_keys = [k for k in results if factor_label in k]
    if not scenario_keys:
        continue

    num_lines = len(BASE_NET.line)
    bar_width = 0.25

    plt.figure(figsize=(5,3))
    x = np.arange(num_lines)

    for i, key in enumerate(scenario_keys):
        wind_bus = key.split('|')[0].strip()
        plt.bar(x + i*bar_width,
                results[key]["line_loading"],
                width=bar_width,
                label=wind_bus)

    # Only one limit line
    plt.axhline(100, color='r', linestyle='--', label="Limit")

    plt.xlabel("Line Index")
    plt.ylabel("Line Loading (%)")
    plt.title(f"Line Loading - {factor_label}")  # Include demand increase
    plt.xticks(x + bar_width*(len(scenario_keys)-1)/2, [f"Line {i}" for i in range(num_lines)])
    plt.legend(loc='upper left', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()


# -------------------------
# Plot transformer loading per load scenario (compact)
# -------------------------
for factor_label in LOAD_FACTORS.keys():
    scenario_keys = [k for k in results if factor_label in k]
    if not scenario_keys:
        continue

    num_trafo = len(BASE_NET.trafo)
    bar_width = 0.25

    plt.figure(figsize=(5,3))
    x = np.arange(num_trafo)

    for i, key in enumerate(scenario_keys):
        wind_bus = key.split('|')[0].strip()
        plt.bar(x + i*bar_width,
                results[key]["trafo_loading"],
                width=bar_width,
                label=wind_bus)

    # Only one limit line
    plt.axhline(100, color='r', linestyle='--', label="Limit")

    plt.xlabel("Transformer Index")
    plt.ylabel("Transformer Loading (%)")
    plt.title(f"Transformer Loading - {factor_label}")  # Include demand increase
    plt.xticks(x + bar_width*(len(scenario_keys)-1)/2, [f"Trafo {i}" for i in range(num_trafo)])

    # Legend slightly lower inside axes
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9), fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()
    
    # -------------------------
# Calculate and plot branch (line) active power losses per load scenario
# -------------------------
for factor_label in LOAD_FACTORS.keys():
    scenario_keys = [k for k in results if factor_label in k]
    if not scenario_keys:
        continue

    num_lines = len(BASE_NET.line)
    bar_width = 0.25

    plt.figure(figsize=(5,3))
    x = np.arange(num_lines)

    for i, key in enumerate(scenario_keys):
        wind_bus = key.split('|')[0].strip()
        # Calculate line active power losses: P_loss = P_from + P_to (both MW)
        # Use res_line.P_from_mw + res_line.P_to_mw
        P_from = BASE_NET.res_line.p_from_mw.values if "p_from_mw" in BASE_NET.res_line else results[key]["line_loading"]*0  # fallback
        P_to = BASE_NET.res_line.p_to_mw.values if "p_to_mw" in BASE_NET.res_line else results[key]["line_loading"]*0  # fallback
        
        # Better: compute losses from net.res_line.p_from_mw + p_to_mw
        # Actually, we need to run a power flow to get these losses
        # Let's compute from net results
        net = copy.deepcopy(BASE_NET)
        net.trafo.at[WIND_TRAFO_IDX, "hv_bus"] = int(wind_bus.split()[-1])
        net.load["p_mw"] *= LOAD_FACTORS[factor_label]
        net.load["q_mvar"] *= LOAD_FACTORS[factor_label]
        try:
            pp.runpp(net)
        except pp.powerflow.LoadflowNotConverged:
            continue
        losses = net.res_line.pl_mw.values  # Pandapower stores line active power losses in MW

        plt.bar(x + i*bar_width, losses, width=bar_width, label=wind_bus)

    plt.xlabel("Line Index")
    plt.ylabel("Active Power Loss (MW)")
    plt.title(f"Line Active Power Losses - {factor_label}")
    plt.xticks(x + bar_width*(len(scenario_keys)-1)/2, [f"Line {i}" for i in range(num_lines)])
    plt.legend(loc='upper left', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()