# -*- coding: utf-8 -*-
"""
Run non-crit ESN and non-crit RBM.

To-do:
    - make sure no numerical instbilities. Potential issues:
        ESN filter giving overflow
        RBM FE too negative

"""

import numpy as np
import pickle
import os
from rbm_connected_myenv_final import ising
from esn_modified import ESN,identity
import string

# load states
file = 'states\\states_cost10_newenv.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']
# quickly change the indices
states['state'] = states.index 
lett = [i for i in string.ascii_uppercase]
states.index=[lett.index(i) for i in states.index.tolist()]
#
## identify best esn parameters
#
#file = 'runData\\esn\\train_esn_newenv.pkl'
#
#with open(file, 'rb') as inp:
#    perfs = pickle.load(inp) 
#
#bestpars = perfs[perfs[:,4] == np.min(perfs[:,4])]
#
#
## ----------------------------------------------------------- set esn
#
#predictor_inputs=2
#predictor_outputs=2 
#predictor_reservoir = int(bestpars[0,0])
#predictor_radius = bestpars[0,1]
#predictor_sparsity = bestpars[0,2]
#input_weights_scaling = bestpars[0,3]
#out_activation=identity
#inverse_out_activation=identity
#input_bias=1
#
##predictor
#predictor = ESN(n_inputs = predictor_inputs,
#                n_outputs = predictor_outputs,
#                n_reservoir=predictor_reservoir,
#                 spectral_radius=predictor_radius,
#                 sparsity=predictor_sparsity,
#                 input_weights_scaling=input_weights_scaling,
#                 out_activation=out_activation, inverse_out_activation=inverse_out_activation,
#                 input_bias=input_bias
#                 )
#
##------------------------------------------------------------ set actor
#
#Ssize = 7
#Msize = 3
#Nhidden = 20
#Nmemory = predictor_reservoir
#size=Ssize+Nmemory+Nhidden+Msize
#
#
##actor
#I = ising(size,Nmemory,Ssize,Msize,predictor)
#I.connectToEnv(states)
#
#
##------------------------------------------------------------- SARSA RL
#
#total_episodes = 10000   #10000
#max_steps = 100         
##lamb = 0.9 # L-2 regularisation parameter
##lr = 0.01
#Beta = np.linspace(0,10,total_episodes)
#
#I.SarsaLearning(total_episodes, max_steps, Beta)
#
##perf measure: track states
##fetable=I.FEtable
##log=I.log
#
#
##----------------------------------------------------------- save
#run = {}
#run['pred_reservoir'] = predictor_reservoir
#run['pred_radius'] = predictor_radius
#run['pred_sparsity'] = predictor_sparsity
#run['pred_scaling'] = input_weights_scaling
#run['pred_out_act'] = out_activation
#run['pred_inv_out_act'] = inverse_out_activation
#run['pred_input_bias'] = input_bias
##
#run['pred_RLSfilter'] = predictor.RLSfilter
#run['pred_W'] = predictor.W
#run['pred_bias'] = predictor.reservoir_bias
#run['pred_W_in'] = predictor.W_in
#run['pred_hist'] = predictor.history
##
#run['i_hidden'] = I.Hsize
#run['total_ep'] = total_episodes
#run['beta'] = Beta
#run['max_steps'] = max_steps
#run['i_hidsen'] = I.hidsen
#run['i_hidmot'] = I.hidmot
#run['i_h'] = I.h
#run['i_fe'] = I.FEtable
#run['t_log'] = I.log
#
#filetosave = 'runData\\rbm\\noncrit_rbm_noncrit_esn_1.pkl'
#
## Save
#if not os.path.isfile(filetosave):
#    with open(filetosave, 'wb') as output:
#            pickle.dump(run, output, pickle.HIGHEST_PROTOCOL)
#else:
#    print("File already exists!")
#
