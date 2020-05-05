# -*- coding: utf-8 -*-
"""

-Look at data param and log...
To-do:
    -create a method in the network which will step through te env. without learning


"""
import pickle
# !!!!!!! mind which ising you import
from rbm_sole_myenv import ising

file_run1 = 'runData\\rbmsole_myenv_run_1_data.pkl'

with open(file_run1, 'rb') as inp:
    run1 = pickle.load(inp)
    
# rbm params
Ssize=run1['Ssize']
Msize=run1['Msize']
Nhidden=run1['Nhidden']
size=run1['size']

#actor
I = ising(size,Ssize,Msize)
I.J = run1['I_J']
I.h = run1['I_h']

# SARSA RL
total_episodes = run1['total_episodes']  
max_steps = run1['max_steps']  
beta = run1['beta']  

#

file_run1 = 'runData\\rbmsolemyenv_cost10hierch0_data.pkl'

with open(file_run1, 'rb') as inp:
    run1 = pickle.load(inp)

log=run1['I_log']
