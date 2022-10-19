# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:45:19 2020

@author: Niki
"""

#
import pickle
import os
from lettersEnv_2 import LettersEnv
import numpy as np
from esn_modified import ESN,identity
from sklearn.model_selection import train_test_split
import time
import string


# return valid number of time steps (disregard the empty time-steps after the episode was terminated)
def valid_steps(episode_data):
#    episode_data[] = 0
#    runs_acc = np.ma.masked_equal(runs_acc, float('inf'))
    return episode_data[episode_data[:,0] != float('inf'),:]


# ============================== Load states

# load states file
file = 'states\\states_cost10_newenv.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']
# quickly change the indices
states['state'] = states.index 
lett = [i for i in string.ascii_uppercase]
states.index=[lett.index(i) for i in states.index.tolist()]


# ============================ Create environment
#create env object
env = LettersEnv(states=states)  



#=============================  Load training samples=
file = 'trainingSamples\\envsampleslincost10_newenv_2.pkl'

with open(file, 'rb') as inp:
    samples = pickle.load(inp) 
    
# !!! dont forget some time steps are infinity!!!
    
    
    
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

# depending on results, do more fine-grained after
n_ress = [10,20,50]
spec_rads = [0.7,0.8,0.9]
spars = [0.7,0.9]
inp_scals = [0.5,0.8,1]
# !!!! to test
#inr=0
#nr=n_ress[0]
#isr=0
#sr=spec_rads[0]
#isp=0
#sp=spars[0]
#iinps=0
#inps=inp_scals[0]

filenm = 'runData\\esn\\train_esn_newenv.pkl'

##==========================================================  Train ESN
comb = 0
ncombs = len(n_ress)*len(spec_rads)*len(spars)*len(inp_scals)
perfs = np.tile(np.repeat(float('inf'), int(5)), (ncombs,1))
for inr,nr in enumerate(n_ress):
    for isr, sr in enumerate(spec_rads):
        for isp, sp in enumerate(spars):
            for iinps, inps in enumerate(inp_scals):
                
                tic=time.perf_counter()
                print('Running comb '+ str(comb) + ' out of ' + str(ncombs))
                
                num_nets = 10
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
                        epinputs = valid_steps(tr_inputs[ep,:,:])
                        epoutputs = valid_steps(tr_outputs[ep,:,:])                        
                        acts = esn.get_states(epinputs, extended=True, continuation=False)
                        epoutputs = esn.inverse_out_activation(epoutputs)
                        for actval,outval in zip(acts,epoutputs):
                            esn.RLSfilter.process_datum(actval.reshape(-1,1), outval.reshape(-1,1))               
                    print('Testing...')
                    preds = np.zeros((te_inputs.shape[0],1))
                    for teep in range(te_inputs.shape[0]):
                        epinputs = valid_steps(te_inputs[teep,:,:])
                        epoutputs = valid_steps(te_outputs[teep,:,:])
                        predsep=np.zeros(epoutputs.shape)
                        acts = esn.get_states(epinputs, extended=True, continuation=False)
                        for iact, actval in enumerate(acts):
                            predsep[iact,:]=esn.out_activation(esn.RLSfilter.predict(actval.reshape(-1,1)).T)
                        preds[teep] = np.mean(np.sum(np.abs(predsep-epoutputs),axis=1))
                    totalmeanerr = np.round(np.mean(preds),2)
                    perfs_networks[irun] = totalmeanerr 
                # calc mean network error
                perfs_networks[np.isnan(perfs_networks)] = 0
                perfs_networks[perfs_networks > 100] = 0
                ma_perfs = np.ma.masked_equal(perfs_networks, 0)
                meannet = np.mean(ma_perfs)
                perfs[comb,:]=[nr,sr,sp,inps,meannet]
##                perfs[comb,:]=[nr,sr,sp,inps,totalmeanerr]
                
                if comb%5 == 0:
                     with open(filenm, 'wb') as output:
                         pickle.dump(perfs, output, pickle.HIGHEST_PROTOCOL)
                         print('Saved perfs up until here...')
                
                comb+=1
                toc=time.perf_counter()
                print('Finished comb '+ str(comb) + ' in ' + str(int((toc - tic)/60)) + ' minutes.')

#save
            
with open(filenm, 'wb') as output:
        pickle.dump(perfs, output, pickle.HIGHEST_PROTOCOL)
                 