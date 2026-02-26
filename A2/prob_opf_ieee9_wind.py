import pandapower as pp, pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
import numpy as np
import random
numba = False

net = pp.from_excel('./A2/ieee9-wind.xlsx')

# --------------------------------------------------------------------
# Step 1: initialize data:
# --------------------------------------------------------------------

# Parameters of Weibull distribution of wind speed
k = 2.02 #Shape parameter [p.u].
lambda_ = 11 #Scale parameter [m/s]

# Parameters of wind power plant
Pwpp = 180 #Max. MW of the wind power plant
# Pwt=2 #Rated MW of individual wind generator.
wsin = 3 # Cut-in wind speed [m/s]
wsr = 12 # Rated wind speed [m/s]
wsout = 20 # Cut-off wind speed [m/s]
wpcoshphi = 1 # Operation at unit power factor 

# --------------------------------------------------------------------
# Step 2: Generate random variation of load and wind power
# --------------------------------------------------------------------

N=100 # Amount of random samples to be generated
np.random.seed(5489) # fix random seed

# Wind power
Pwpp_act = []
ws_rand = np.asarray([random.weibullvariate(lambda_, k) for i in range(N)]) #Random wind speed following Weibull distribution
#Pwpp_act=np.zeros(len(ws_rand))
for i in ws_rand:
    if i<wsin:              # P = 0
        Pwpp_act.append(0)
    elif i>wsin and i<wsr:  # partial load
        Pwpp_act.append(Pwpp*(i**3-wsin**3)/(wsr**3-wsin**3))
    elif i>wsr and i<wsout: # full load
        Pwpp_act.append(Pwpp)
    elif i>wsout:           # P = 0
        Pwpp_act.append(0)

# Array of wind powers
Pwpp_act=np.asarray(Pwpp_act)

# Active-reactive power (pq) ratio of loads
means = np.asarray(net.load.p_mw)
sd = np.random.uniform(1,30,3)
pq_ratio = np.asarray((net.load.q_mvar/net.load.p_mw))  # Power Factor (PF)

# --------------------------------------------------------------------
# Step 3: run the probabilistic power flow (PPF)
# --------------------------------------------------------------------

for p in Pwpp_act:
    # Create random load variations with normal distribution
    net.load.p_mw = np.random.normal(means, sd)
    net.load.q_mvar = np.asarray(net.load.p_mw) * pq_ratio
    
    # Set wind powers
    net.sgen.p_mw = p
    net.sgen.max_q_mvar = p * np.tan(np.arccos(wpcoshphi))
    net.sgen.min_q_mvar = -p * np.tan(np.arccos(wpcoshphi))
    net.sgen.max_p_mw = p
    net.sgen.min_p_mw = p
    
    # Run optimal power flow:
    # Note: not every sample operating condition (associated to random sampled 
    # values of load demand and wind power output) leads to a convergent OPF. 
    # Hence we use try: and except:
    try: 
        pp.runpp(net)
        pp.runopp(net, init='pf',verbose=False)
        
        # Store results
        plt.figure(1)
        plt.scatter(net.bus.index, net.res_bus.vm_pu, c='blue')
        plt.xlabel("Bus Index")
        plt.ylabel("Voltage (pu)")
        plt.figure(2)
        plt.scatter(net.line.index, net.res_line.loading_percent, c = 'green')
        plt.xlabel("Line Index")
        plt.ylabel("Loading (%)")
        
    except:
        continue

plt.show()


