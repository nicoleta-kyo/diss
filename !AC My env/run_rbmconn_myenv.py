# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:01:14 2020

@author: Niki
"""

import numpy as np
import pandas as pd
from rbm_modified_myenv import ising
from esn_modified import ESN


def identity(x):
    return x

def letttonum(abc_array):
    newarr = [None] * len(abc_array)
    for i in range(len(abc_array)):
        abc = abc_array[i]
        new = ''
        for char in abc:
            if char == 'A':
                new+='0'
            elif char == 'B':
                new+='1'
            elif char == 'C':
                new+='2'
            elif char == 'D':
                new+='3'
        newarr[i] = new
    
    return newarr   

def create_ex_states():
    
    st = [['A', 1], ['B', 1], ['C', 1], ['D', 1]]
    states = pd.DataFrame(st, columns = ['state', 'cost'])
    states['comb'] = ['A', 'B', 'C', 'D']
    states['bitseq'] = states['comb']
    
    df2 = pd.DataFrame([['T1', 2, 'AC', 'AC'],
                        ['Q', 2, 'CB', 'CB'],
                        ['G', 3, 'T1B', 'ACB'],
                        ['F', 2, 'CC', 'CC'],
                        ['P', 2, 'CA', 'CA'],
                        ['Y', 5, 'GF', 'ACBCC'],
                        ['Q', 6, 'YA', 'ACBCCA']
                        ], columns = ['state', 'cost', 'comb', 'bitseq'])
    states=states.append(df2)
    states.reset_index(drop=True, inplace=True)
    # add int bitzeq
    states['bitseqint'] = letttonum(states.bitseq.values)
    max_cost = 6

    return states, max_cost

# create example states
    
states, max_cost = create_ex_states()
goal = states.index.values[-1]

# esn params
predictor_inputs = 2 # linear state,action
predictor_outputs = 2 # linear state2, reward
predictor_reservoir = 12
predictor_radius = 0.9
predictor_sparsity = 0.9
out_activation = identity

#predictor
predictor = ESN(n_inputs = predictor_inputs, n_outputs = predictor_outputs, n_reservoir=predictor_reservoir,
                 spectral_radius=predictor_radius, sparsity=predictor_sparsity,
                 noise=0.001,
                 readout='pseudo-inverse',
                 ridge_reg=None,
                 input_weights_scaling = 1,
                 out_activation=out_activation, inverse_out_activation=out_activation
                 )

# rbm params
Ssize=7
Msize=4
Nhidden = 20
Nmemory = predictor_reservoir
size=Ssize+Nmemory+Nhidden+Msize

#actor
I = ising(size,Nmemory,Ssize,Msize,predictor)
I.connectToEnv(states, max_cost, goal)

# SARSA RL
total_episodes = 1000   #10000
max_steps = 50         # in Otsuka they are 250 but it's a different task
Beta=None

I.SarsaLearning(total_episodes, max_steps, Beta)
I.displayRunData(total_episodes, 7)

#
