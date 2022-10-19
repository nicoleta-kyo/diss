# -*- coding: utf-8 -*-
"""
- Train the ESN on the goal sequence (of states version 1).
    An episode is 10 steps and it always reaches the goal state (receiving reward=1) at the end.
    No episodic collection of the state activations (the network does not start from a zero state at every episode.)


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

## =============================================== Create training samples
numeps=500
goalseq=[0,1,3,1,3,3,1,3,3,3]
numsteps=len(goalseq)


samples=np.zeros((numeps*numsteps,4))
# collect env data (without resetting)
t=0
for i in range(numeps):
    if i % 1000 == 0:
        print(i)
    s = env.reset(numsteps)
    for a in range(numsteps):
        s2, r, done = env.step(goalseq[a])
        ## !! try to only do action as input and state as output
        samples[t, :] = [s, goalseq[a], s2, r]
        s=s2
        t+=1

#========================================================= Separate train and test
# I wanna keep the order because it's correct sequences
split=3000
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
n_ress = [20, 100, 200] #[10,20,50,100,200]
spec_rads = [0.7, 0.9] #[0.7,0.8,0.9,1]
spars = [0.7, 0.9]  #[0.7,0.8,0.9]
inp_scals =  [0.5, 0.7, 0.9, 1]    #[0.5,0.8,0.9,1]
#!!!!! to test
#inr=0
#nr=20
#isr=0
#sr=0.9
#isp=0
#sp=0.9
#iinps=0
#inps=1


#==========================================================  Train ESN
comb = 0
ncombs = len(n_ress)*len(spec_rads)*len(spars)*len(inp_scals)
num_nets= 10
elements_tosave = 5
perfs = np.tile(np.repeat(float('inf'), elements_tosave), (ncombs,1))      # store parameter values and performance
perfs_networks = np.tile(np.repeat(float('inf'), 4 + num_nets), (ncombs,1)) 
for inr,nr in enumerate(n_ress):
    for isr, sr in enumerate(spec_rads):
        for isp, sp in enumerate(spars):
            for iinps, inps in enumerate(inp_scals):
                
                tic=time.perf_counter()
                print('Running comb '+ str(comb) + ' out of ' + str(ncombs))

                perfs_networks[comb,:4] = [nr,sr,sp,inps]
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
                                  input_bias=input_bias
                                  )
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
#                    if np.isnan(trainerr):
#                        raise ValueError('Train error in network '+str(irun) + ' is NaN.')
                        
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
#                    if np.isnan(testerr):
#                        raise ValueError('Test error in network '+str(irun) + ' is NaN.')
#                    if testerr > 100:
#                        raise ValueError('Test error is > 100. Investigate.')
                    perfs_networks[comb,int(4+irun)] = testerr
                
                # mask the numerically instable filters
                ma_perfs = perfs_networks[comb,4:]
                ma_perfs[np.isnan(ma_perfs)] = 0
                ma_perfs[ma_perfs > 100] = 0
                ma_perfs = np.ma.masked_equal(ma_perfs, 0)
                # calc mean network error
                meannet = np.mean(ma_perfs)
                perfs[comb,:]=[nr,sr,sp,inps,meannet]
                
                comb+=1
                toc=time.perf_counter()
                print('Finished comb '+ str(comb) + ' in ' + str(int((toc - tic)/60)) + ' minutes.')
                
#                
###### Save results 
                
filenm = 'runData\\esn\\train_soleesn_cost10_lin_goalseq.pkl'

if not os.path.isfile(filenm):
    with open(filenm, 'wb') as output:
        pickle.dump(perfs, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")              