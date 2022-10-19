
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:01:14 2020

@author: Niki
"""

import numpy as np
import pandas as pd
from rbm_sole_myenv__changingFE import ising
import pickle
import os


#----------------------------------------------- RUN

# rbm params
Ssize=7
Msize=3
Nhidden = 20
size=Ssize+Nhidden+Msize

#  load states file
file = 'states\\states_max_cost10_hierch.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']

#actor
I = ising(size,Ssize,Msize)
I.connectToEnv(states)

# SARSA RL
total_episodes = 10000   #10000
max_steps = 10         # lets do 50 as well
beta=np.repeat(1, total_episodes) # will be default - 1

I.SarsaLearning(total_episodes, max_steps, beta)
fet = I.FEtable
np.sum(fet[:,51,0])

"""
-- Running...
 - 250/200 steps- the actor always ends up reaching the goal state
 - now running 100 to see if it does....
         it always reaches it! strange... look at the results you ran befoe to see....
 - if I don't show all states reached at a time step, the network doesn't know it's reached the rest (and it might be useful)
     I'm really not sure. Meditate on this.
      need to think how to encode it also??
 - another idea: showing no state instead of the bases? If I am to do that, a state would be only the memory. will take a bit of time
"""

# the currect weights and biases of the
steps=50
I.env.reset(steps)
for i in range(steps):
    I.env

