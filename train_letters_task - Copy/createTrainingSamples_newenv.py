# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:43:04 2020

@author: Niki
"""


import pickle
import os
from lettersEnv_2 import LettersEnv
import numpy as np
import math
import numpy.random as nprand
from esn_modified import ESN,identity,MAE,atanh
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pdb
import string



# ============================== Load states

# load states file
file = 'states\\states_cost10_newenv.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']
# quickly change the indices
states['state'] = states.index 
lett = [i for i in string.ascii_uppercase]
states.index=[lett.index(i) for i in states.index.tolist()]


#create env object
env = LettersEnv(states=states)
numeps = 2000  
# !
#


# =============================================== Create training samples
numsteps = 200
samples=np.tile(np.repeat(float('inf'),4),(numeps,numsteps, 1))
for epis in range(numeps):
    if epis % 100 == 0:
            print('Episode '+ str(epis))
    s = env.reset(numsteps)
    for step in range(numsteps):
        a = nprand.choice(env.num_act)
        s2, r, done = env.step(a)
        samples[epis, step, :] = [s, a, s2, r]
        s=s2
        if done:
            break
        
# save the samples
filenm = 'envsampleslincost10_newenv_2.pkl'

if not os.path.isfile(filenm):
    with open(filenm, 'wb') as output:
        pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!") 

# Load samples=
file = 'trainingSamples\\envsampleslincost10_newenv_2.pkl'

with open(file, 'rb') as inp:
    samp = pickle.load(inp) 
#      
    