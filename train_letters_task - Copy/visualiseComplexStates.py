# -*- coding: utf-8 -*-
"""

Create heat-map for the complex states combinations

"""

import pickle
import numpy as np
import string


def createHeatMapComplexStates(dicstates,maptitle,imgname):
    
    import matplotlib.pyplot as plt
    
    # build array for heatmap
    combs = np.zeros((14,14),dtype='int')
    for k in dicstates.keys():
        for ke in dicstates[k]:
            combs[ke,k] = round(dicstates[k][ke],2)*100
    
    
    start_c=4
    end_c=13
    start_s=7
    end_s=14
    combs1=combs[start_c:end_c,start_s:end_s]
#    statesticks=[i for i in range(start_s,end_s)]
#    comsticks=[i for i in range(start_c,end_c)]
    stlabs=[string.ascii_uppercase[i] for i in range(start_s,end_s)]
    combslabs=[string.ascii_uppercase[i] for i in range(start_c,end_c)]
    
    fig, ax = plt.subplots()
    im = ax.imshow(combs1)
    
    ax.set_xticks(np.arange(len(stlabs)))
    ax.set_yticks(np.arange(len(combslabs)))
    ax.set_xticklabels(stlabs)
    ax.set_yticklabels(combslabs)
    ax.set_xlabel('state')
    ax.set_ylabel('sub-state')
    
    ax.set_title(maptitle)
    
#    # Create colorbar
#    cbar = ax.figure.colorbar(im, ax=ax)
#    cbar.ax.set_ylabel('Percentage combinations covered', rotation=-90, va="bottom")
    
    for i in range(len(combslabs)):
        for j in range(len(stlabs)):
            if combs1[i, j] != 0 or (i == 8 and j == 6):
                text = ax.text(j, i, combs1[i, j],
                               ha="center", va="center", color="w")
    fig.tight_layout()
    fig1=plt.gcf()
    fig1.savefig('runData\\res\\'+imgname)
    
    return im

# -- import

file = 'runData\\res\\perf_AcritPnon_compSt.pkl'

with open(file, 'rb') as input:
    critrbmresdata = pickle.load(input)

compst = critrbmresdata['complexStates_new']
title='$A^* P$'
fname='complex_states_AcritPnon_new'

#compst[7]=complexStates[7]

# -- run

createHeatMapComplexStates(compst,title, fname)
