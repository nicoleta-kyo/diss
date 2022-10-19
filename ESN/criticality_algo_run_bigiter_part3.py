# -*- coding: utf-8 -*-
"""
Determine criticality

"""

import pickle
import os
import numpy as np
from generate_narma import gen_narma
#from ttictoc import TicToc
from itertools import product
from criticality_algo_functions import det_criticality

#=================== Params
    
# filename for run if to save!!!!!!!!!!
filenm = 'criticality_run_bigiter_part3.pkl'



# ================= For Network (not being optimised)
n_inputs=1
n_outputs=1
n_reservoir = 100
aug = True
trans = 200

esn_config = {}
esn_config['n_inputs'] = n_inputs
esn_config['n_outputs'] = n_outputs
esn_config['n_reservoir'] = n_reservoir
esn_config['augmented'] = aug
esn_config['transient'] = trans



#================= For NARMA signal

#params 
#order = 10
#size = 1200
#coef_u = 0.4    # instead of 1.5 because of overflow error
#lb_input = 0
#ub_input = 1

# generate i/o signal
#u = np.random.uniform(low = lb_input, high = ub_input, size = size)
#narma_in = u[:-1].reshape(len(u)-1,1)
#narma_out = gen_narma(u, order, size, coef_u)[1:]

file_load = 'criticality_run1.pkl'

with open(file_load, 'rb') as input:
    config_run1 = pickle.load(input)
# i/o signal
narma_in = config_run1['narma_in']
narma_out = config_run1['narma_out']




# ================ For Param Space

# define param space: spectral_radius, sparsity, input_scaling
# and create combinations of the different parameter values

spectral_radius = np.round(np.linspace(start=0.4, stop=1.6, num=10),2)
sparsity = np.round(np.linspace(start=0.3, stop=0.9, num=10),2)
scaling = np.round(np.linspace(start=0.3, stop=0.8, num=10),2)
param_space = np.round(list(product(spectral_radius, sparsity, scaling)),2)

param_space = param_space[500:]
#

d = param_space.shape[1]    # number of parameters

sigma_p = 0.5   # sd of the perturbations




# =================== For number of runs

num_iter = 499     #number of iterations to do in the algorithm! ~ stopping criterion
                    # change it to dynamically determine the number of combinations
num_trials = 10    # number of trials
num_pert = 80 # number of perturbations



# ==================== Run

#t = TicToc()
#t.tic()

determinants, crit_params = det_criticality(esn_config, narma_in, narma_out, param_space, num_iter, num_trials, num_pert, sigma_p)

#t.toc()
#print(t.elapsed)






# ==================== Save objects in a dictionary and save to file

# params for narma
narmap = {}
narmap['order'] = 10
narmap['size'] = 1200
narmap['coef_u'] = 1.5
narmap['lb_input'] = 0
narmap['ub_input'] = 0.5

#params for run
run_objects = {}
run_objects['esn_config'] = esn_config
run_objects['narma_in'] = narma_in
run_objects['narma_out'] = narma_out
run_objects['narma_params'] = narmap
run_objects['param_space'] = param_space
run_objects['sigma_p'] = sigma_p
run_objects['num_iter'] = num_iter
run_objects['num_trials'] = num_trials
run_objects['num_pert'] = num_pert
#output
run_objects['determinants'] = determinants
run_objects['best_param_set'] = crit_params


if not os.path.isfile(filenm):
    with open(filenm, 'wb') as output:
        pickle.dump(run_objects, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")