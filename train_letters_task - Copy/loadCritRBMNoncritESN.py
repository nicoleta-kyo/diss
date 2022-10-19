      # -*- coding: utf-8 -*-
"""

Load crit RBM and non-crit ESN results
and calculate performance measures

To-do's:
    - plot number of states over episodes
    - create heat-map for complex states
    - plot predictor quality over episodes
    - plot internal reward over episodes


"""

import pickle
import numpy as np
from perfMeasurePassedCombs import checkNumbitsBeforeN
import string
import os


# --------------------------------------------------- Crit RBM - non-crit ESN results


file = 'runData\\rbm\\critical_rbm_esn.pkl'

with open(file, 'rb') as input:
    critrbm = pickle.load(input)
    
eps = 70
hist = critrbm['pred_history'][:eps]
log = critrbm['pred_log'][:eps]

#i wanna count reached number of states per every 200 steps
# the env does not reset when the goal is reached.
hist = hist.reshape(700000,4)
hist=hist.reshape(3500,200,4)
# get rif of infs from first episode
hist = hist[hist[:,0,0]!=float('inf')]
hist=hist.astype(int)

# same for log
log2 = log.reshape(700000,2)
log3=log2.reshape(3500,200,2)
# get rif of infs from first episode
log4 = log3[log3[:,0,0]!=-float('inf')]


#---------------------= load states

file = 'states\\states_cost10_newenv.pkl'

with open(file, 'rb') as inp:
    std = pickle.load(inp) 
states = std['states']
# quickly change the indices
states['state'] = states.index 
lett = [i for i in string.ascii_uppercase]
states.index=[lett.index(i) for i in states.index.tolist()]


#--------------------------------  I. 1.a number of states reached

## states reached per every 200 steps!
#neweps=hist4.shape[0]
#rst2=np.zeros((neweps,1))
#for i in range(neweps):
# uniques = np.unique(hist4[i,:,2])
# rst2[i] = len(uniques)
# 
##average visited states
#vis_st = np.floor(np.mean(rst2))
#
## for plot
#vals = len([i for i in range(0,neweps,50)])
#xticks=[]  
#plot_vis_st1 = np.zeros(vals)
#for i in range(vals):
#    if i==(vals-1):
#        plot_vis_st1[i] = np.mean(rst2[50*i,:3455])
#    plot_vis_st1[i] = np.mean(rst2[50*i,:50*(i+1)])
#    xticks.append(50*i)


# 1.a' - number of combs per state reached?
 
 
#----------------------------------------I. 1.b did it go through all combs until reaching a state?

complexStates = {}
for i in range(5,14):
    print(i)
    complexStates[i] = checkNumbitsBeforeN(hist, states, i)




#----------------------------------------I. 2. how many time steps was goal state reached?
eps_goal = np.where(hist[:,:,2] == 13)[0]
ntgoal = len(np.where(hist[:,:,2] == 13)[0])
# total time-steps/10 because this is the length of the action sequence needed to get to that state
totsteps = hist.shape[0]*hist.shape[1]/10
# times goal reached over the number of 10-step time sequences
pg = ntgoal/totsteps





#----------------------------------------II. mean pred quality per episode and mean internal reward
predq = np.mean(log4,axis=1)


#
#
##------------------------------- Save data
#res={}
#res['num_st'] = vis_st
#res['hierch_combs'] = complexStates
#res['goal'] = pg
#res['pred_qual'] = predq
#
#filetosave = 'runData\\res\\perf_AcritPnon.pkl'
#
#if not os.path.isfile(filetosave):
#    with open(filetosave, 'wb') as output:
#            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
#else:
#    print("File already exists!")
#
 
res = {}
res['complexStates_new'] = complexStates
    
filetosave = 'runData\\res\\perf_AcritPnon_compSt.pkl'

if not os.path.isfile(filetosave):
    with open(filetosave, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")

"""

Future work/discussion:
    - here both internal and external reward are used.
        -Internal reward is really small - would have tried to use a bigger one to see if it changes the results.
        -?should I do only internal (only if I have time)
    - comment on predictor quality:
        -It does not seem to improve - maybe because of the way the environment is created.
        Maybe if it only returned complex states, not the bases, it would perform better.

    
"""
