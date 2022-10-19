# -*- coding: utf-8 -*-
"""
- Collect action sequences from the states matrix, and use them as training samples for the network.


"""

import pickle
import os
from lettersEnv import LettersEnv
import numpy as np
import numpy.random as nprand
from esn_modified import ESN,identity,MAE
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

# load states file
file = 'states\\states_max_cost10_hierch.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']

#create env object
env = LettersEnv(states=states)

# =============================================== Create training samples
numseqs=3000
acts=[]
for i in range(numseqs):
    seq = nprand.choice(env.num_combs)
    acts+=[int(i) for i in states.bitseqint[seq]]
    
# collect env data (without resetting)
samples = np.zeros((len(acts),4))
s=-1
env.reset(len(acts))  # reset in order to create the ssm array
for i,a in enumerate(acts):
    if i % 1000 == 0:
        print(i)
    s2, r, done = env.step(a)
    ## !! try to only do action as input and state as output
    samples[i, :] = [s, a, s2, r]
    s=s2

#========================================================= Separate train and test
# I wanna keep the order because it's correct sequences
split=6500
tr_inputs = samples[:split,:2]
tr_outputs = samples[:split,2:]
te_inputs = samples[split:,:2]
te_outputs = samples[split:,2:]

#== Params which remain unchanged
n_inputs=2
n_outputs=2           
out_activation=identity
inverse_out_activation=identity
input_bias=1

#== Grid search
n_ress = [20,200]
spec_rads = [0.5,0.9]
spars = [0.7,0.9]
inp_scals = [0.7,1]
#!!!!! to test
#inr=0
#nr=20
#isr=0
#sr=0.7
#isp=0
#sp=0.7
#iinps=0
#inps=1


#==========================================================  Train ESN
comb = 0
ncombs = len(n_ress)*len(spec_rads)*len(spars)*len(inp_scals)
perfs = np.tile(np.repeat(float('inf'), 6), (ncombs,1))      # store parameter values and performance
for inr,nr in enumerate(n_ress):
    for isr, sr in enumerate(spec_rads):
        for isp, sp in enumerate(spars):
            for iinps, inps in enumerate(inp_scals):
                
                tic=time.perf_counter()
                print('Running comb '+ str(comb) + ' out of ' + str(ncombs))
                
                n_reservoir=nr
                spectral_radius=sr
                sparsity=sp
                input_weights_scaling = inps
                
                num_nets=5
                perfs_networks=np.tile(np.repeat(float('inf'),2),(num_nets, 1))
                for irun in range(num_nets):
                    print('Instantiating network ' + str(irun))
                    
                    esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs,
                                  n_reservoir=n_reservoir
                                  spectral_radius=spectral_radius, sparsity=sparsity,
                                  input_weights_scaling = input_weights_scaling,
                                  out_activation=out_activation, inverse_out_activation=inverse_out_activation,
                                  input_bias=input_bias)
                    print('RLS training...')
                    acts = esn.get_states(tr_inputs, extended=True, continuation=False)
                    for t in range(acts.shape[0]):
                        act = acts[t,:].reshape(-1,1)
                        trout = esn.inverse_out_activation(tr_outputs[t,:].reshape((1, -1))).T
                        esn.RLSfilter.process_datum(act, trout)               
                    # train error
                    trpreds=np.zeros(tr_outputs.shape)
                    for t in range(acts.shape[0]):
                        act=acts[t,:].reshape(-1,1)
                        trpreds[t,:] = esn.out_activation(esn.RLSfilter.predict(act).T)
                    res = np.mean(np.abs(trpreds - tr_outputs),axis=1)
                    trainerr=np.mean(res)
                    
#                    # plot tr error
#                    trans=8000
#                    x = np.arange(res.shape[0]-trans)
#                    y = res[trans:]
#                    l1 = 'tr error'
#                
#                    fig, ax = plt.subplots()
#                    line1, = ax.plot(x, y, label=l1)
#                    ax.legend()
#                    plt.show()
#                    
                    
                    print('Testing...')
                    preds=np.zeros(te_outputs.shape)
                    acts = esn.get_states(te_inputs, extended=True, continuation=False)
                    for t in range(acts.shape[0]):
                        act=acts[t,:].reshape(-1,1)
                        preds[t,:] = esn.out_activation(esn.RLSfilter.predict(act).T)
                    # calc mean per time step (for all episodes)     
                    meants = np.mean(np.abs(preds - te_outputs),axis=1)
                    # calc total error averaging the time steps
                    testerr = np.mean(meants)
                    if np.isnan(testerr):
                        break
                    perfs_networks[irun,:] = [trainerr,testerr]
                    
                # calc mean network error
                meannet = np.mean(perfs_networks,axis=0)
                perfs[comb,:]=np.hstack([nr,sr,sp,inps,meannet])
##                perfs[comb,:]=[nr,sr,sp,inps,totalmeanerr]
                
                comb+=1
                toc=time.perf_counter()
                print('Finished comb '+ str(comb) + ' in ' + str(int((toc - tic)/60)) + ' minutes.')
                
                
##### Save results
                
filenm = 'train_soleesn_cost10_lin_existingseq.pkl'

if not os.path.isfile(filenm):
    with open(filenm, 'wb') as output:
        pickle.dump(perfs, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")              