import pandapower as pp, pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

<<<<<<< HEAD
net = pp.from_excel('./A2/ieee9-wind.xlsx')
=======
net = pp.from_excel(
    r'C:/Users/dagma/Documents/Master/SET3065 Intelligent Electrical Power Grids/IEPG-SET3065-main/A2/ieee9-wind.xlsx'
)
>>>>>>> c225c97ab70b895946a0a1f42e63a0787808b213

# Running the optimization power flow problem
pp.runopp(net, init='pf',verbose=False)
print(f"The total cost is: {net.res_cost}.")

# Plot bus voltages
fig, ax1 = plt.subplots(1,1)
voltages = net.res_bus.vm_pu.to_list()
ax1.bar(range(len(voltages)), voltages)
ax1.set_xlabel("Bus Index")
ax1.set_ylabel("Voltage (pu)")
plt.show()

# TO DO: plot line losses 
line_losses = net.res_line.loading_percent.to_list()
fig, ax2 = plt.subplots(1,1)
ax2.bar(range(len(line_losses)), line_losses)
ax2.set_xlabel("Line Index")
ax2.set_ylabel("Line Losses (%)")
plt.show()