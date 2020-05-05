# -*- coding: utf-8 -*-
"""
Train ESN with binary env inputs and outputs
activation function - tanh


"""


import pickle
from lettersEnv import LettersEnv
import numpy as np
import math
import numpy.random as nprand
from esn_modified import ESN,identity,MAE,atanh
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pdb


# Transform positive integer into bit array
def bitfield(n, size):
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)

# Transform env input to bit array
def inputToBit(x, xmax, bitsize):
    ind = int(np.floor((x + xmax) / (2 * xmax + 10 * np.finfo(float).eps) * 2**bitsize))
    x = [int(x) for x in bin(int(ind))[2:]]
    x = [0] * (bitsize - len(x)) + x
    return np.array(x)

def binthresh(x):
    if x.ndim <2:
        x=x.reshape(1,-1)
    binx=np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        binx[i]=0 if x[0,i] < 0.5 else 1
        
    return np.array(binx,dtype=int)
    
        


# ============================== Load states

# load states file
file = 'states\\states_max_cost10_hierch.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']

#create env object
env = LettersEnv(states=states)
numeps = 10    #2000

# =============================================== Create training samples
numsteps = 50    # reward not received once
stsize = 7
actsize = 3
rewsize = 2
nio = int(stsize*2+actsize+rewsize)  # number of input-output elements
samples=np.tile(np.repeat(-1,nio),(numeps,numsteps, 1))
for epis in range(numeps):
    if epis % 100 == 0:
            print('Episode '+ str(epis))
    env.reset(numsteps)
    s=-1
    for step in range(numsteps):
        a = nprand.choice(env.num_act)
        s2, r, done = env.step(a)
        sb = inputToBit(s, env.num_combs, stsize)
        s2b = inputToBit(s2, env.num_combs, stsize)
        ab = inputToBit(a, env.num_act, actsize)
        rb = inputToBit(r, 2, rewsize)          # external reward is either 0 or 1
        samples[epis, step, :] = [*sb, *ab, *s2b, *rb]
        s=s2
        if done:
            break


#========================================================= Params

# params which remain unchanged
n_inputs=stsize+actsize
n_outputs=stsize+rewsize       
out_activation=np.tanh
inverse_out_activation=atanh
input_bias=1

#== Grid search
n_ress = [100,200]
spec_rads = [0.7,0.8,0.9,1]
spars = [0.7,0.8,0.9]
inp_scals = [0.5,0.8,1]
# !!!! to test
#n_ress = [100]
#spec_rads = [0.7]
#spars = [0.7]
#inp_scals = [0.5]
#inr=0
#nr=n_ress[0]
#isr=0
#sr=spec_rads[0]
#isp=0
#sp=spars[0]
#iinps=0
#inps=inp_scals[0]


#========================================================= Separate train and test
train, test = train_test_split(samples, test_size=0.2)
tr_inputs = train[:,:,:n_inputs]
tr_outputs = train[:,:,n_inputs:]
te_inputs = test[:,:,:n_inputs]
te_outputs = test[:,:,n_inputs:]


#==========================================================  Train ESN
comb = 0
ncombs = len(n_ress)*len(spec_rads)*len(spars)*len(inp_scals)
perfs = np.tile(np.repeat(float('inf'), 5), (ncombs,1))      # store parameter values and performance
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
                perfs_networks=np.repeat(float('inf'),num_nets)
                for irun in range(num_nets):
                    print('Instantiating network ' + str(irun))
                    
                    esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs,
                                 spectral_radius=spectral_radius, sparsity=sparsity,
                                 input_weights_scaling = input_weights_scaling,
                                 out_activation=out_activation, inverse_out_activation=inverse_out_activation,
                                 input_bias=input_bias)
                    print('RLS training...')
                    for ep in range(tr_inputs.shape[0]):
                        epinputs = tr_inputs[ep,:,:]
                        epoutputs = tr_outputs[ep,:,:]
                        acts = esn.get_states(epinputs, extended=True, continuation=False)
                        for actval,outval in zip(acts,epoutputs):
                            teach = np.round(esn.inverse_out_activation(outval.reshape((1, -1))),2)
                            esn.RLSfilter.process_datum(actval.reshape(-1,1), teach.reshape(-1,1))               
                    print('Testing...')
                    preds=np.zeros(te_outputs.shape, dtype=int)
                    for teep in range(te_inputs.shape[0]):
                        epinputs = te_inputs[teep,:,:]
                        acts = esn.get_states(epinputs, extended=True, continuation=False)
                        for iact, actval in enumerate(acts):
                            preds[teep,iact,:] = binthresh(esn.out_activation(esn.RLSfilter.predict(actval.reshape(-1,1)).T))
                    # calc mean per time step (for all episodes)     
                    meants = np.mean(np.sum(np.abs(preds - te_outputs),axis=2),axis=0)
                    # calc total error averaging the time steps
                    totalmeanerr = np.mean(meants)
                    perfs_networks[irun] = totalmeanerr 
                # calc mean network error
                meannet = np.mean(perfs_networks)
                perfs[comb,:]=[nr,sr,sp,inps,meannet]
##                perfs[comb,:]=[nr,sr,sp,inps,totalmeanerr]
                
                comb+=1
                toc=time.perf_counter()
                print('Finished comb '+ str(comb) + ' in ' + str(int((toc - tic)/60)) + ' minutes.')

# 22:12
                
import os

filenm = 'train_soleesn_cost10_binary.pkl'

if not os.path.isfile(filenm):
    with open(filenm, 'wb') as output:
        pickle.dump(perfs, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")              