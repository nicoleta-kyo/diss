
# -*- coding: utf-8 -*-
"""
Sole RBM on MyEnv



"""


import numpy as np
import pandas as pd
from rbm_sole_myenv import ising
import pickle
import os
import time


def identity(x):
    return x


#----------------------------------------------- RUN

# run name - will add index for every param combo
runnm = 'runData\\rbmsolemyenv_cost25hierch'

# load states file
file = 'states\\states_max_cost25_hierch.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']

# to optimise
eps = [5000, 10000]
steps = [50,100,200]
hids= [20,40]
#!!!!!  to test script
#eps = [3,4]
#steps = [5,6,7]
#hids = [20,20]
#
betas=['s', 'inc']
lrts=['s','inc']
ncp = len(eps)*len(steps)*len(hids)*len(betas)*len(lrts)


i = 0
for ie,e in enumerate(eps):
    for iss,s in enumerate(steps):
        for ih,h in enumerate(hids):
            for ib,b in enumerate(betas):
                for ilr,lr in enumerate(lrts):
                    
                    tic = time.perf_counter()
                    # !!!!! add printing
                    saveAs = runnm+str(i)
                    pltnm = 'e='+str(e)+', '+'s='+str(s)+', '+'h='+str(h)+', '+'b='+str(b)+', '+'lr='+str(lr)
                    
                    # rbm params
                    Ssize=7
                    Msize=4
                    Nhidden=h
                    size=Ssize+Nhidden+Msize
                    
                    #actor
                    I = ising(size,Ssize,Msize)
                    I.connectToEnv(states)
                    
                    # SARSA RL
                    total_episodes = e   #10000
                    max_steps = s         # lets do 50 as well
                    beta=np.repeat(1,total_episodes) if b == 's' else np.linspace(1,15,total_episodes) # will be default - 1
                    learn_rate = np.repeat(0.01,total_episodes) if lr == 's' else np.linspace(0.1,0.01,total_episodes)
                    
                    I.SarsaLearning(total_episodes, max_steps, beta, learn_rate)
                    I.displayRunData(display=False, save=True,savePath=saveAs, plotname = pltnm)
                    
                    #perf
                    # get sum of max rewards/episodes
                    perf = I.getPerf()
                    
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
                    params['I_log'] = I.log
                    params['perf'] = perf
                    
                    filenmparams = saveAs + '_data.pkl'
                    
                    if not os.path.isfile(filenmparams):
                        with open(filenmparams, 'wb') as output:
                                pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)
                    else:
                        print("File already exists!")
                        
                    i+=1
                    
                    toc = time.perf_counter()
                    print('Combination ' + str(i) + ' out of '+ str(ncp) + ' took ' + str(int((toc - tic)/60)) + ' minutes.')

