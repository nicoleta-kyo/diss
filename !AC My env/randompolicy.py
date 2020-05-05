# -*- coding: utf-8 -*-
"""
Use random policy to collect samples for training the esn
Train the esn with linear inputs-outputs with RLS for uncentered data, no regularisation
- cost 10; 50 time steps

To-do:
-for time-steps 100,200
-binary inputs-otputs
-regularisation?

"""
import pickle
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
numeps = 2000

# =============================================== Create training samples
numsteps = 50    # reward not received once
samples=np.tile(np.repeat(-2,4),(numeps,numsteps, 1))
for epis in range(numeps):
    env.reset(numsteps)
    s=-1
    for step in range(numsteps):
        a = nprand.choice(env.num_act)
        s2, r, done = env.step(a)
        samples[epis, step, :] = [s, a, s2, r]
        s=s2
        if done:
            break

#numsteps = 100    # running for 100 to see if reward received
#samples100 = np.tile(np.repeat(-2,4),(numeps,numsteps, 1))
#for epis in range(numeps):
#    env.reset(numsteps)
#    s=-1
#    for step in range(numsteps):
#        a = nprand.choice(env.num_act)
#        s2, r, done = env.step(a)
#        samples100[epis, step, :] = [s, a, s2, r]
#        s=s2
#        if done:
#            break

#========================================================= Separate train and test
train, test = train_test_split(samples, test_size=0.2)
tr_inputs = train[:,:,:2]
tr_outputs = train[:,:,2:]
te_inputs = test[:,:,:2]
te_outputs = test[:,:,2:]
            
# !!! to test
#tr_inputs = tr_inputs[:5,:,:]
#tr_outputs = tr_outputs[:5,:,:]
#te_inputs = te_inputs[:5,:,:]
#te_outputs = te_outputs[:5,:,:]

#== Params which remain unchanged
n_inputs=2
n_outputs=2           
out_activation=identity
inverse_out_activation=identity
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

#==========================================================  Train ESN
comb = 0
ncombs = len(n_ress)*len(spec_rads)*len(spars)*len(inp_scals)
perfs = np.tile(np.repeat(float('inf'), int(4+50)), (ncombs,1))
for inr,nr in enumerate(n_ress):
    for isr, sr in enumerate(spec_rads):
        for isp, sp in enumerate(spars):
            for iinps, inps in enumerate(inp_scals):
                
                perfs_networks=np.repeat(-1,10)
                
                tic=time.perf_counter()
                print('Running comb '+ str(comb) + ' out of ' + str(ncombs))
                
                n_reservoir=nr
                spectral_radius=sr
                sparsity=sp
                input_weights_scaling = inps
                
                for irun in range(5):
                    print('Instantiating network ' + str(irun))
                    
                    esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs,
                                 spectral_radius=spectral_radius, sparsity=sparsity,
                                 input_weights_scaling = input_weights_scaling,
                                 out_activation=identity, inverse_out_activation=identity,
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
                    meanep = np.round(np.mean(np.sum(np.abs(preds - te_outputs),axis=2),axis=0),2)
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

