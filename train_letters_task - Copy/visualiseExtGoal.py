# -*- coding: utf-8 -*-
"""

Create a plot for the crit rbm's two versions and a plot for the non-crit rbm's two versions
for predictor's goal state coverage

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

def calc_mean_qual(eps, qual):
    vals = len([i for i in range(0,eps,50)])  
    quals = np.zeros(vals)
    for i in range(vals):
        if i==(vals-1):
            quals[i] = np.mean(qual[50*i:eps])
        quals[i] = np.mean(qual[50*i:50*(i+1)])
    return quals

#---------------------------load
    
file = 'runData\\res\\perf_AnonPnon.pkl'

with open(file, 'rb') as input:
    data = pickle.load(input)

goal_anonpnon = data['goal_ep']

    
file = 'runData\\res\\perf_AnonPcrit.pkl'

with open(file, 'rb') as input:
    data = pickle.load(input)

goal_anonpcrit = data['goal_ep']


file = 'runData\\res\\perf_AcritPnon_goal.pkl'

with open(file, 'rb') as input:
    data = pickle.load(input)

goal_acritpnon = data['goal_eps']

file = 'runData\\res\\perf_AcritPcrit_goal.pkl'

with open(file, 'rb') as input:
    data = pickle.load(input)

goal_acritpcrit = data['goal_eps']


# ---------- Plot predictor quality

# --------- Crit RBM
eps1 = 5453
x1=range(eps1)
#x1 = [i for i in range(0,eps1,50)]
#xticks1=[i for i in range(0,eps1,500)]

l1 = 'A-crit P-non-crit'
y1 = np.zeros(eps1)
for i in range(eps1):
    y1[i] = 1 if i in goal_acritpnon else 0
l2 = 'A-crit P-crit'
y2 = np.zeros(eps1)
for i in range(eps1):
    y2[i] = 2 if i in goal_acritpcrit else 0
#
# --------- Non-crit RBM
eps2 = 5453
x2=range(eps2)

l3 = 'A-non-crit P-non-crit'
y3 = np.zeros(eps2)
for i in range(eps2):
    y3[i] = 3 if i in goal_anonpnon else 0
l4 = 'A-non-crit P-crit'
y4 = np.zeros(eps2)
for i in range(eps2):
    y4[i] = 4 if i in goal_anonpcrit else 0

# -------- Plot

# crit actor models
fig, ax = plt.subplots()
line1, = ax.plot(x2, y1, color='black',marker='.',linestyle='',label=l1)
line2, = ax.plot(x2, y2, color='dimgrey',marker='.',linestyle='', label=l2)
line3, = ax.plot(x2, y3, color='darkgrey',marker='.',linestyle='', label=l3)
line4, = ax.plot(x2, y4, color='silver',marker='.',linestyle='', label=l4)
#ax.legend()
ax.set_ylim([0.1,5])

ax.set_xlabel('episode')
ax.set_yticklabels(['','$A^* P$', '$A^* P^*$','$A P$','$A P^*$'])

fig1 = plt.gcf()
fig1.savefig('runData\\res\\ext_goal')
