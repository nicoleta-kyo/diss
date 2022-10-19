

# -*- coding: utf-8 -*-
"""
Run the ESN with NARMA input with parameters, number of runs and error measure outlined in section 3 of
     https://papers.nips.cc/paper/2318-adaptive-nonlinear-system-identification-with-echo-state-networks.pdf

"""

import numpy as np
import statistics as st
from esn import ESN, atanh
from generate_narma import gen_narma
import pickle
import math
#import warnings
#warnings.filterwarnings("error")




# ============ Network configuration and hyperparameter values ============

config = {}

# Input config

config['input_signal_type'] = 'NARMA'# input signal

# esn config
 
config['n_inputs'] = 1 # nr of input dimensions
config['n_outputs'] = 1 # nr of output dimensions
config['n_reservoir'] = 100 # nr of reservoir neurons
config['spectral_radius'] = 0.8 # spectral radius of the recurrent weight matrix
config['sparsity'] =  0.95 #  proportion of recurrent weights set to zero
config['noise'] = 0.0001 # noise added to each neuron (regularization) 
config['readout'] = 'pseudo-inverse'     # readout : can be ridge or pseudo-inverse
config['input_weights_scaling'] = 1
config['out_activation'] = np.tanh # output activation function (applied to the readout) 
config['inverse_out_activation'] = atanh # inverse of the output activation function 
config['silent'] = True # supress messages
config['augmented'] = True  # whether to use augmented states array or not
config['transient'] = 200   # washout


# error to use
config["error"] = 'NMSE'    # mse, nmse or rnmse

# ======== Set lists of different order parameters for NARMA ========
    
        
order = 10
coef_u = 1.5
lb_input = 0
ub_input = 0.5
#size = np.repeat(1200,4)
size_tr = 1200
size_te = 2200


runs_signal = 10
perf_all = np.zeros(runs_signal)
 # ============ Load/create input ============
for rs in range(runs_signal):
    
    #create input signals
    u_tr = np.random.uniform(low = lb_input, high = ub_input, size = size_tr)
    d_train = u_tr[:-1]
    d_train = np.reshape(d_train, (len(d_train), 1))
    #output - without first element - so u and d align (otherwise d(n+1) = u(n) + ...)
    d_tr_out = gen_narma(u_tr, order, size_tr, coef_u)[1:]
    while (d_tr_out >= 1).any():
        u_tr = np.random.uniform(low = lb_input, high = ub_input, size = size_tr)
        d_train = u_tr[:-1]
        d_train = np.reshape(d_train, (len(d_train), 1))
        #output - without first element - so u and d align (otherwise d(n+1) = u(n) + ...)
        d_tr_out = gen_narma(u_tr, order, size_tr, coef_u)[1:]
    
    u_te = np.random.uniform(low = lb_input, high = ub_input, size = size_te)
    d_test = u_te[:-1]
    d_test = np.reshape(d_test, (len(d_test), 1))
    d_te_out = gen_narma(u_te, order, size_te, coef_u)[1:]
    while (d_te_out >= 1).any():
        u_te = np.random.uniform(low = lb_input, high = ub_input, size = size_te)
        d_test = u_te[:-1]
        d_test = np.reshape(d_test, (len(d_test), 1))
        d_te_out = gen_narma(u_te, order, size_te, coef_u)[1:]
    
    runs = 100
    runs_acc = np.zeros(runs)
    # do 50 runs
    for run in range(runs):
        
        print("Run " + str(run) + "...")
    
            
        # ============ Specify, train and evaluate model ============
        
        esn = ESN(
                n_inputs=config['n_inputs'],
                n_outputs=config['n_outputs'],
                n_reservoir=config['n_reservoir'],
                spectral_radius=config['spectral_radius'],
                sparsity=config['sparsity'],
                noise=config['noise'],
                
                readout = config['readout'],
                
                input_weights_scaling=config['input_weights_scaling'],
                
                out_activation=config['out_activation'],
                inverse_out_activation=config['inverse_out_activation'],
                
                silent = config['silent'],
                augmented = config['augmented'],
                transient = config['transient']
                )
        
        # train
        pred_train = esn.fit(d_train, d_tr_out, inspect=False)
        # test
        pred_test = esn.predict(d_test, continuation=False)
        #print("NARMA order: " + str(order[i]))
        
        # [nk] discard the transient when calculating the error
        # calculate mse
        mse = np.mean((pred_test[config['transient']:] - d_te_out[config['transient']:])**2)
        
        # calculate nmse
        var = st.pvariance(d_te_out[config['transient']:,0])
        nmse = mse/var
        rnmse = np.sqrt(nmse)
        
        #print("testing error: " + str(te_err))
        if config['error'] == 'MSE':
            round_acc = mse
        elif config['error'] == 'NMSE':
            round_acc = nmse
        elif config['error'] == 'RNMSE':
            round_acc = rnmse
        else:
            raise ValueError("Invalid error function")
        
    
        #add results to the run
        runs_acc[run] = round_acc
    
    perf_all[rs] = np.mean(runs_acc)

print(np.mean(perf_all))

#warnings.filterwarnings("default")

def perc_change(new, old):
    return (new-old)/old*100






