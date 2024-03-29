# -*- coding: utf-8 -*-
"""
- Run critical RBM on my env. 


"""

import numpy as np
import pdb
import os
import pickle
from esn_modified import ESN,identity
from rbm_critical_myenv import ising
import string

 

#============================================== load states file

file = 'states\\states_cost10_newenv.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']
# quickly change the indices
states['state'] = states.index 
lett = [i for i in string.ascii_uppercase]
states.index=[lett.index(i) for i in states.index.tolist()]



#============================================= Critical ESN

# identify best esn parameters

file1out = 'runData\\esn\\critical_esn_myenv_output.pkl'
with open(file1out, 'rb') as input:
    critout1 = pickle.load(input)
file2out = 'runData\\esn\\critical_esn_myenv_output_run2.pkl'
with open(file2out, 'rb') as input:
    critout2 = pickle.load(input)   

dets1=critout1['dets']
np.max(dets1)
    
dets2=critout2['dets']
np.max(dets2) 

file1param = 'runData\\esn\\critical_esn_myenv_params.pkl'

with open(file1param, 'rb') as input:
    critpar1 = pickle.load(input)
    
bestpars = critout1['param_set_star']
critpar = critpar1['esn_config']

predictor_inputs = 2 # linear state,action
predictor_outputs = 2 # linear state2, reward
predictor_reservoir = critpar['n_reservoir']
predictor_radius = float(bestpars[0,0])
predictor_sparsity = float(bestpars[1,0])
scaling = float(bestpars[2,0])
out_act = critpar['out_activation']
inv_out_act = critpar['inverse_out_activation']
bias = critpar['input_bias']

predictor = ESN(n_inputs = predictor_inputs,
                n_outputs = predictor_outputs,
                n_reservoir=predictor_reservoir,
                 spectral_radius=predictor_radius,
                 sparsity=predictor_sparsity,
                 input_weights_scaling = scaling,
                 out_activation=out_act,
                 inverse_out_activation=inv_out_act,
                input_bias = bias
                 )


# ============================================ RBM

# rbm params
Ssize=7
Msize=3
Nhidden = 50        # 20,50,60?
Nmemory = predictor_reservoir
Nreward = 4
size=Ssize+Nmemory+Nhidden+Nreward+Msize
# for training
# did 700 iterations only!!!
Iterations = 700                          # originally 10000
T = 1000                                   # originally 5000

I = ising(size,Nmemory,Nreward,Ssize,Msize,predictor,states,T)

# Import reference correlations
filecorr = 'correlations-ising2D-size400.npy'
Cdist = np.load(filecorr)

# reorder reference correlations to match network's current correlations
I.m1 = np.zeros(size)
I.Cint = np.zeros((size, size - 1))
for i in range(size):
    c = []
    for j in range(size - 1):
        ind = np.random.randint(len(Cdist))
        c += [Cdist[ind]]
    I.Cint[i, :] = -np.sort(-np.array(c))
    

# Train
I.CriticalLearning(Iterations, T)

# states reached
# np.unique(I.predictor.history[0,:,2])
# np.unique(I.predictor.history[2,:,2])

# pred quality

# np.round(I.log[2,I.log[2,:] > 0.1],2)


# Params to save
params = {}
params['Ssize']= Ssize
params['Msize']= Msize
params['nhidden'] = Nhidden
params['size'] = size
params['Iterations'] = Iterations
params['T'] = T
params['network_J'] = I.J
params['network_h'] = I.h
params['pred_history'] = I.predictor.history
params['pred_log'] = I.log
# esn params
params['pred_reservoir'] = predictor_reservoir
params['pred_radius '] = predictor_radius 
params['pred_sparsity'] = predictor_sparsity
params['pred_scaling'] = scaling
params['pred_out_act'] = out_act
params['pred_inv_out_act'] = inv_out_act
params['pred_bias'] = bias


# Name of the file to save run

filetosave = 'runData\\critical_rbm_critical_esn.pkl'

# Save
if not os.path.isfile(filetosave):
    with open(filetosave, 'wb') as output:
            pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")