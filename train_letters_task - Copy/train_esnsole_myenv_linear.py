# -*- coding: utf-8 -*-
"""
- Train the ESN on randomly generated actions.
    At every episode the environment resets, and the network starts collecting activations from a zero state.

"""


import pickle
import os
from lettersEnv import LettersEnv
import numpy as np
import math
import numpy.random as nprand
from esn_modified import ESN,identity,MAE,atanh
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pdb



# ============================== Load states

# load states file
file = 'states\\states_max_cost10_hierch.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']

#create env object
env = LettersEnv(states=states)
numeps = 2000  


# ============================ Load training data

file = 'trainingSamples\\envsamples_cost10_linear.pkl'

with open(file, 'rb') as inp:
    samples = pickle.load(inp)
    
#==================================Separate train and test
    
train, test = train_test_split(samples, test_size=0.2)
tr_inputs = train[:,:,:2]
tr_outputs = train[:,:,2:]
te_inputs = test[:,:,:2]
te_outputs = test[:,:,2:]

# =============================== Params

#== Params which remain unchanged
n_inputs=2
n_outputs=2           
out_activation=identity
inverse_out_activation=identity
input_bias=1

#== Grid search
n_ress = [20, 100, 200] #[10,20,50,100,200]
spec_rads = [0.7, 0.9]#[0.7,0.8,0.9,1]
spars =  [0.7, 0.9]  #[0.7,0.8,0.9]
inp_scals = [0.7, 0.9]    #[0.5,0.8,0.9,1]
# !!!! to test
#inr=0
#nr=n_ress[0]
#isr=0
#sr=spec_rads[0]
#isp=0
#sp=spars[0]
#iinps=0
#inps=inp_scals[0]

#==========================================================  Train ESN
comb = 0
ncombs = len(n_ress)*len(spec_rads)*len(spars)*len(inp_scals)
perfs = np.tile(np.repeat(float('inf'), int(5)), (ncombs,1))
for inr,nr in enumerate(n_ress):
    for isr, sr in enumerate(spec_rads):
        for isp, sp in enumerate(spars):
            for iinps, inps in enumerate(inp_scals):
                
                tic=time.perf_counter()
                print('Running comb '+ str(comb) + ' out of ' + str(ncombs))
                
                num_nets = 5
                perfs_networks=np.repeat(float('inf'),num_nets)                
                for irun in range(num_nets):
                    print('Instantiating network ' + str(irun))
                    
                    esn = ESN(n_inputs=n_inputs,
                              n_outputs=n_outputs,
                                 n_reservoir=nr,
                                 spectral_radius=sr,
                                 sparsity=sp,
                                 input_weights_scaling = inps,
                                 out_activation=out_activation,
                                 inverse_out_activation=inverse_out_activation,
                                 input_bias=input_bias)
                    print('RLS training...')
                    for ep in range(tr_inputs.shape[0]):
                        epinputs = tr_inputs[ep,:,:]
                        epoutputs = tr_outputs[ep,:,:]
                        acts = esn.get_states(epinputs, extended=True, continuation=False)
                        epoutputs = esn.inverse_out_activation(epoutputs)
                        for actval,outval in zip(acts,epoutputs):
                            esn.RLSfilter.process_datum(actval.reshape(-1,1), outval.reshape(-1,1))               
                    print('Testing...')
                    preds=np.zeros(te_inputs.shape)
                    for teep in range(te_inputs.shape[0]):
                        epinputs = te_inputs[teep,:,:]
                        acts = esn.get_states(epinputs, extended=True, continuation=False)
                        for iact, actval in enumerate(acts):
                            preds[teep,iact,:]=esn.out_activation(esn.RLSfilter.predict(actval.reshape(-1,1)).T)
                    # calc mean per time step (over episodes)     
                    meanep = np.mean(np.sum(np.abs(preds - te_outputs),axis=2),axis=0)
                    # calc total error averaging the time steps
                    totalmeanerr = np.mean(meanep)
                    perfs_networks[irun] = totalmeanerr 
                # calc mean network error
                meannet = np.mean(perfs_networks)
                perfs[comb,:]=[nr,sr,sp,inps,meannet]
##                perfs[comb,:]=[nr,sr,sp,inps,totalmeanerr]
                
                comb+=1
                toc=time.perf_counter()
                print('Finished comb '+ str(comb) + ' in ' + str(int((toc - tic)/60)) + ' minutes.')
