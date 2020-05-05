# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:01:14 2020

@author: Niki
"""

import numpy as np
import pandas as pd
from rbm_sole_myenv import ising
import pickle
import os


#----------------------------------------------- RUN

runI = 1    #index for saving files
saveAs = 'runData\\rbmsole_myenv_run_'+str(runI)

# rbm params
Ssize=7
Msize=4
Nhidden = 20
size=Ssize+Nhidden+Msize

num_states = 10 
max_cost = 30
goal=None  # will be max cost state

#actor
I = ising(size,Ssize,Msize)
I.connectToEnv(num_states, max_cost, goal)

# SARSA RL
total_episodes = 10000   #10000
max_steps = 100         # lets do 50 as well
beta=None # will be default - 1


I.SarsaLearning(total_episodes, max_steps, beta)
I.displayRunData(total_episodes,save=True,savePath=saveAs)

#
#runData to save - the plots, the run params, I.h and I.J

params = {}
params['Ssize'] = Ssize
params['Msize'] = Msize
params['Nhidden'] = Nhidden
params['size'] = size
params['total_episodes'] = total_episodes
params['max_steps'] = max_steps
params['beta'] = beta
params['I_h'] = I.h
params['I_J'] = I.J
params['I_log'] = np.log

filenmparams = saveAs + '_data.pkl'

if not os.path.isfile(filenmparams):
    with open(filenmparams, 'wb') as output:
            pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")



