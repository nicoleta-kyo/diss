# -*- coding: utf-8 -*-
"""

Create a plot for the crit rbm's two versions and a plot for the non-crit rbm's two versions
for nummber of states reached.

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


# ---------- Plot number of visited states

# --------- Crit RBM
eps1 = 3455
x1 = [i for i in range(0,eps1,50)]
xticks1=[i for i in range(0,eps1,500)]

y1 = plot_vis_st1
l1 = 'A-crit P-non-crit'
y2 = plot_vis_st2
l2 = 'A-crit P-crit'

# --------- Non-crit RBM
eps2 = 5453
x2 = [i for i in range(0,eps2,50)]
xticks2=[i for i in range(0,eps2,500)]

y3 = plot_vis_st3
l3 = 'A-non-crit P-non-crit'
y4 = plot_vis_st4
l4 = 'A-non-crit P-crit'

# -------- Plot

plotname = 'number of visited states'
saveP = ''
    
fig, ax = plt.subplots()
line1, = ax.plot(x1, y1, 'k-',label=l1)
line2, = ax.plot(x1, y2, 'k-.',label=l2)
ax.legend()
ax.set_ylim([0,14])
ax.set_xlabel('episode')
ax.set_ylabel('number of reached states')

fig1 = plt.gcf()
fig1.savefig('runData\\res\\num_states_Acrit')


fig, ax = plt.subplots()
line1, = ax.plot(x2, y3, 'k-', label=l3)
line2, = ax.plot(x2, y4, 'k-.', label=l4)
ax.legend()
ax.set_ylim([0,14])
ax.set_xlabel('episode')
ax.set_ylabel('number of reached states')

fig1 = plt.gcf()
fig1.savefig('runData\\res\\num_states_Anoncrit')
