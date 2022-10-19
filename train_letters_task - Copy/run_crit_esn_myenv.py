# -*- coding: utf-8 -*-
"""

-- Determine criticality for ESN on my environment.

"""

import pickle
import os
import numpy as np
from itertools import product
from criticality_algo_functions_modified import det_criticality
from esn_modified import identity

#=================== Params
    
# filename for run if to save!!!!!!!!!!
filenmparams = 'runData\\esn\\critical_esn_myenv_params.pkl'
filenmoutput = 'runData\\esn\\critical_esn_myenv_output.pkl'

   

# ================= For Network (not being optimised)
n_inputs = 2
n_outputs = 2
n_reservoir = 20  #!!!  for now = might need to run for more         
out_activation=identity
inverse_out_activation=identity
input_bias=1

esn_config = {}
esn_config['n_inputs'] = n_inputs
esn_config['n_outputs'] = n_outputs
esn_config['n_reservoir'] = n_reservoir
esn_config['out_activation'] = out_activation
esn_config['inverse_out_activation'] = inverse_out_activation
esn_config['input_bias'] = input_bias


#================= Load training samples

file_load = 'trainingSamples\\envsampleslincost10_newenv_2.pkl'

with open(file_load, 'rb') as input:
    samples = pickle.load(input)

inputs = samples[:500,:,:2]
#  dont forget some time steps are infinity!!! - handled in the crit algo method!
# take only 500 episodes for now and see how long it will run

# ================ For Param Space

# define param space: spectral_radius, sparsity, input_scaling
# and create combinations of the different parameter values

spectral_radius = np.round(np.linspace(start=0.4, stop=1.6, num=10),2)
sparsity = np.round(np.linspace(start=0.3, stop=0.9, num=10),2)
scaling = np.round(np.linspace(start=0.3, stop=0.8, num=10),2)
param_space = np.round(list(product(spectral_radius, sparsity, scaling)),2)

param_space = param_space[:500]
#

d = param_space.shape[1]    # number of parameters
sigma_p = 0.5   # sd of the perturbations




# =================== For number of runs

num_iter = 500    # for now cover half of the params (radius > 1 gives bad results ingrid search)
num_trials = 5    # number of trials
num_pert = 20 # number of perturbations


# =================== Save params to file.


#params for run
run_objects = {}
run_objects['esn_config'] = esn_config
run_objects['inputs'] = inputs
run_objects['param_space'] = param_space
run_objects['sigma_p'] = sigma_p
run_objects['num_iter'] = num_iter
run_objects['num_trials'] = num_trials
run_objects['num_pert'] = num_pert


# 
if not os.path.isfile(filenmparams):
    with open(filenmparams, 'wb') as output:
            pickle.dump(run_objects, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")




# =========== Run - it will dynamically rewrite the output with determinants and param sets to the filenmoutput file

det_criticality(esn_config, inputs, param_space, num_iter, num_trials, num_pert, sigma_p, filenmoutput)


