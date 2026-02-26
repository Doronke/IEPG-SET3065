import pandapower as pp, pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

net = pp.from_excel(
    r'C:/Users/dagma/Documents/Master/SET3065 Intelligent Electrical Power Grids/IEPG-SET3065-main/A2/ieee9-wind.xlsx'
)

# Running the optimization power flow problem
pp.runopp(net, init='pf',verbose=False)
print(f"The total cost is: {net.res_cost}.")

# Plot bus voltages
fig, ax1 = plt.subplots(1,1)
voltages = net.res_bus.vm_pu.to_list()
ax1.bar(range(len(voltages)), voltages)
ax1.set_xlabel("Bus Index")
ax1.set_ylabel("Voltage (pu)")

# TO DO: plot line losses 
