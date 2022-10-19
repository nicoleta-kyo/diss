# -*- coding: utf-8 -*-
"""
Run the ESN with NARMA input with parameters, number of runs and error measure outlined in section 3 of
     https://papers.nips.cc/paper/2318-adaptive-nonlinear-system-identification-with-echo-state-networks.pdf

"""

import numpy as np
import statistics as st
from esn import ESN
from generate_narma import gen_narma
import pickle
#import warnings
#warnings.filterwarnings("error")

# ============ Load paramater set to evaluate

file_pars = 'C:\\nikyk\\.THIRD YEAR\\1. DISS\\ESN\\cknd-pyESN\\Run results\\criticality_run_bigtrain_part2_params.pkl'
file_out = 'Run results\criticality_run_bigtrain_part2_output.pkl'


with open(file_pars, 'rb') as input:
   new_run_pars = pickle.load(input)
with open(file_out, 'rb') as input:
   new_run_out = pickle.load(input)   
   
#dets_newnarma = new_run['determinants']
best = new_run_out['param_set_star']
narma_params = new_run_pars['narma_params']

param_set_star = best

# ============ Network configuration and hyperparameter values ============

config = {}

# Input config

config['input_signal_type'] = 'NARMA'# input signal

# esn config
 
config['n_inputs'] = 1 # nr of input dimensions
config['n_outputs'] = 1 # nr of output dimensions
config['n_reservoir'] = 100 # nr of reservoir neurons
config['spectral_radius'] = param_set_star[0] # spectral radius of the recurrent weight matrix
config['sparsity'] =  param_set_star[1] #  proportion of recurrent weights set to zero
config['noise'] = 0.0001 # noise added to each neuron (regularization) 
config['readout'] = 'pseudo-inverse'     # readout : can be ridge or pseudo-inverse
config['ridge_reg']  = 0     # regularisation for ridge regression, if 0 = least squares
config['input_weights_scaling'] = param_set_star[2]  # scaling of the input weights
config['out_activation'] = np.tanh # output activation function (applied to the readout) 
config['inverse_out_activation'] = np.arctanh # inverse of the output activation function 
config['silent'] = True # supress messages
config['augmented'] = True  # whether to use augmented states array or not
config['transient'] = 200   # washout


# error to use
config["error"] = 'NMSE'    # mse, nmse or rnmse

# ======== Set lists of different order parameters for NARMA ========
    
# List of orders
#order = [8,9,10,11]
#order = [narma_params['order']]
#coef_u = narma_params['coef_u']
#lb_input = narma_params['lb_input']
#ub_input = narma_params['ub_input']
order = [10]
coef_u = 1.5
lb_input = 0
ub_input = 0.5
#size = np.repeat(1200,4)
size_tr = [1200]
size_te = [2200]

runs = 100
runs_acc = np.zeros((len(order),runs))
# do 50 runs
for run in range(runs):
    
    print("Run " + str(run) + "...")
    
    round_acc = np.zeros((1,len(order)))
    for i in range(len(order)):
        
        # ============ Load/create input ============
        
        #create input signals
        u_tr = np.random.uniform(low = lb_input, high = ub_input, size = size_tr[i])
        u_te = np.random.uniform(low = lb_input, high = ub_input, size = size_te[i])
        
        # create NARMA input and output signals: train and test
        #input - without last element - no output for it
        d_train = u_tr[:-1]
        d_train = np.reshape(d_train, (len(d_train), 1))
        #output - without first element - so u and d align (otherwise d(n+1) = u(n) + ...)
        d_tr_out = gen_narma(u_tr, order[i], size_tr[i], coef_u)[1:]
        
        #input - without last element - no output for it
        d_test = u_te[:-1]
        d_test = np.reshape(d_test, (len(d_test), 1))
        #output - without first element - so u and d align (otherwise d(n+1) = u(n) + ...)
        d_te_out = gen_narma(u_te, order[i], size_te[i], coef_u)[1:]
        
        # if NARMA signal contains inf values, don't use it as input
        if np.isinf(d_train).any() or np.isinf(d_test).any():
            
            break
        
        # ============ Specify, train and evaluate model ============
        
        esn = ESN(
                n_inputs=config['n_inputs'],
                n_outputs=config['n_outputs'],
                n_reservoir=config['n_reservoir'],
                spectral_radius=config['spectral_radius'],
                sparsity=config['sparsity'],
                noise=config['noise'],
                
                readout = config['readout'],
                ridge_reg = config['ridge_reg'],
                
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
            round_acc[:,i] = mse
        elif config['error'] == 'NMSE':
            round_acc[:,i] = nmse
        elif config['error'] == 'RNMSE':
            round_acc[:,i] = rnmse
        else:
            raise ValueError("Invalid error function")
        
    
    #add results to the run
    runs_acc[:,run] = round_acc
    

# Get averages of results
#mask nan values
runs_acc[np.isnan(runs_acc)] = 0
runs_acc = np.ma.masked_equal(runs_acc, 0)
mean_err = np.mean(runs_acc, axis=1)[0]
print("Mean " + config['error'] + ": " + str(mean_err))
print("Minimum " + config['error'] + ": " + str(np.min(runs_acc)))


#warnings.filterwarnings("default")

def perc_change(new, old):
    return (new-old)/old*100












