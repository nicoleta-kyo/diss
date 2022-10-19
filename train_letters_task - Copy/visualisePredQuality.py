# -*- coding: utf-8 -*-
"""

Create a plot for the crit rbm's two versions and a plot for the non-crit rbm's two versions
for predictor's quality/internal reward

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

# --- import data

file = 'runData\\res\\perf_AcritPnon.pkl'
with open(file, 'rb') as input:
    data = pickle.load(input)
q1 = data['pred_qual'][:,1]
file = 'runData\\res\\perf_AcritPcrit.pkl'
with open(file, 'rb') as input:
    data = pickle.load(input)
q2 = data['pred_qual'][:,1]
file = 'runData\\res\\perf_AnonPnon.pkl'
with open(file, 'rb') as input:
    data = pickle.load(input)
q3 = data['']
file = 'runData\\res\\perf_AnonPcrit.pkl'
with open(file, 'rb') as input:
    data = pickle.load(input)
q4 = data['']


# ---------- Plot predictor quality

# --------- Crit RBM
eps1 = 3455
x1=np.arange(0,3455)
x1 = [i for i in range(0,eps1,50)]
xticks1=[i for i in range(0,eps1,500)]

l1 = 'A-crit P-non-crit'
y1 = calc_mean_qual(eps1,q1)
l2 = 'A-crit P-crit'
y2 = calc_mean_qual(eps1,q2)

# --------- Non-crit RBM
eps2 = 5453
x2 = [i for i in range(0,eps2,50)]
xticks2=[i for i in range(0,eps2,500)]


l3 = 'A-non-crit P-non-crit'
y3 = calc_mean_qual(eps2,q3)
l4 = 'A-non-crit P-crit'
y4 = calc_mean_qual(eps2,q4)

# -------- Plot

    # crit actor models
fig, ax = plt.subplots()
line1, = ax.plot(x1, y1, 'k-',label=l1)
line2, = ax.plot(x1, y2, 'k-.',label=l2)
ax.legend()
ax.set_ylim([0,0.01])
ax.set_xlabel('episode')
ax.set_ylabel("internal reward")

fig1 = plt.gcf()
fig1.savefig('runData\\res\\intr_Acrit')


     # non crit actor models
fig, ax = plt.subplots()
line1, = ax.plot(x2, y3, 'k-', label=l3)
line2, = ax.plot(x2, y4, 'k-.', label=l4)
ax.legend()
ax.set_ylim([0,0.01])
ax.set_xlabel('episode')
ax.set_ylabel("internal reward")

fig1 = plt.gcf()
fig1.savefig('runData\\res\\intrq_Anoncrit')
