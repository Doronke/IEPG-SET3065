# conda activate pandapower_env

import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as plot

# ------------------------------
# Load network
# ------------------------------
net = pn.case4gs()

# ------------------------------
# Run original power flow
# ------------------------------
pp.runpp(net)
print("=== Original bus results ===")
print(net.res_bus)

# ------------------------------
# Show network components for diagram
# ------------------------------
print("\n=== Network components (for SLD) ===")
print("\nBuses:")
print(net.bus[['name', 'vn_kv']])

print("\nLines (connections):")
for idx, row in net.line.iterrows():
    print(f"Line {idx}: Bus {row.from_bus} -> Bus {row.to_bus}, {row.length_km} km")

print("\nLoads:")
for idx, row in net.load.iterrows():
    print(f"Load {idx}: Bus {row.bus}, {row.p_mw} MW, {row.q_mvar} MVAr")

print("\nGenerators:")
for idx, row in net.gen.iterrows():
    print(f"Gen {idx}: Bus {row.bus}, {row.p_mw} MW, {row.vm_pu} p.u.")

print("\nExternal grid:")
for idx, row in net.ext_grid.iterrows():
    print(f"Ext Grid {idx}: Bus {row.bus}, {row.vm_pu} p.u.")

# ------------------------------
# Show original line flows
# ------------------------------
line_results = net.res_line.join(net.line[['from_bus', 'to_bus']])
line_results['loss_mw'] = line_results['p_from_mw'] + line_results['p_to_mw']  # line loss in MW
print("\n=== Original line flows ===")
print(line_results[['from_bus','to_bus','p_from_mw','q_from_mvar','p_to_mw','q_to_mvar','loss_mw']])

# ------------------------------
# Reduce load at bus 2 from 200 MW to 80 MW
# ------------------------------
net.load.loc[net.load.bus == 2, 'p_mw'] = 80

# ------------------------------
# Run new power flow
# ------------------------------
pp.runpp(net)
print("\n=== New bus results after load reduction ===")
print(net.res_bus)

# ------------------------------
# Show new line flows
# ------------------------------
line_results_new = net.res_line.join(net.line[['from_bus', 'to_bus']])
line_results_new['loss_mw'] = line_results_new['p_from_mw'] + line_results_new['p_to_mw']
print("\n=== New line flows after load reduction ===")
print(line_results_new[['from_bus','to_bus','p_from_mw','q_from_mvar','p_to_mw','q_to_mvar','loss_mw']])


# ------------------------------
# Optional simple plot (basic layout)
# ------------------------------
plot.simple_plot(net, show_plot=True)