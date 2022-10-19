# -*- coding: utf-8 -*-
"""
Implementation of i/o sequence NARMA from Section 3 in
https://papers.nips.cc/paper/2318-adaptive-nonlinear-system-identification-with-echo-state-networks.pdf

"""

#1.5

import numpy as np
import matplotlib.pyplot as plt

def gen_narma(u, order, size, coef_u):
    d = np.random.uniform(0, 0.5, order)
    d = np.reshape(d, (len(d),1))
    
    if size <= order:
        return d[:size]
    
    for i in range(size-order):
        new = 0.3*d[-1] + 0.05*d[-1]*sum(d[-order:]) + coef_u*u[len(d)-order]*u[len(d)-1] + 0.1
        new = np.reshape(new,(1,1))
        d = np.concatenate((d,new))
        
    return d


#order = 10
#size = 1200
#coef_u = 1.5
#u = np.random.uniform(0, 0.5, size)
#
#d = gen_narma(u, order, size)
#
#plt.plot(d)
#plt.ylabel('NARMA ' + 'order ' + str(order))
#plt.xlabel('n')
#plt.show()


