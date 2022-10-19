
"""
Load non-crit RBM and crit ESN results
and calculate performance measures

"""

import pickle
import numpy as np
from perfMeasurePassedCombs import checkNumbitsBeforeN
import string 
import os


# --------------------------------------------------- Non-crit RBM - crit ESN results


file = 'runData\\rbm\\noncrit_rbm_crit_esn.pkl'

with open(file, 'rb') as input:
    noncritrbm = pickle.load(input)

eps = 5453  
#allst = 14
hist = noncritrbm['pred_hist'][:eps].astype(int)
log = noncritrbm['t_log'][:eps]
# maskinfs
log2 = np.ma.masked_equal(log, -float('inf'))

fetable = noncritrbm['i_fe']
    
    
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

# states reached per every 200 steps!
neweps=hist.shape[0]
rst2=np.zeros((neweps,1))
for i in range(neweps):
 uniques = np.unique(hist[i,:,2])
 rst2[i] = len(uniques)
 
# average visited states
vis_st = np.floor(np.mean(rst2))


# for plot
vals = len([i for i in range(0,neweps,50)])  
plot_vis_st4 = np.zeros(vals)
for i in range(vals):
    if i==(vals-1):
        plot_vis_st4[i] = np.mean(rst2[50*i,:neweps])
    plot_vis_st4[i] = np.mean(rst2[50*i,:50*(i+1)])

# 1.a' - number of combs per state reached?
# 
 
#----------------------------------------I. 1.b did it go through all combs until reaching a state?

complexStates = {}
for i in range(7,14):
    print(i)
    complexStates[i] = checkNumbitsBeforeN(hist, states, i)



##----------------------------------------I. 2. how many time steps was goal state reached?
eps_goal = np.where(hist[:,:,2] == 13)[0]
ntgoal = len(np.where(hist[:,:,2] == 13)[0])
# total time-steps/10 because this is the length of the action sequence needed to get to that state
totsteps = hist.shape[0]*hist.shape[1]/10
# times goal reached over the number of 10-step time sequences
pg = ntgoal/totsteps


#----------------------------------------II. mean pred quality per episode and mean internal reward
predq_ep = np.ma.filled(np.mean(log2[:,:,0],axis=1),0)
intrew_ep = np.ma.filled(np.mean(log2[:,:,1],axis=1),0)


#------------------------------- Save data
res={}
res['num_st'] = vis_st
res['hierch_combs'] = complexStates
res['goal_ep'] = eps_goal
res['goal_mean'] = pg 
res['pred_qual'] = predq_ep
res[''] = intrew_ep

filetosave = 'runData\\res\\perf_AnonPcrit.pkl'

if not os.path.isfile(filetosave):
    with open(filetosave, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")
#
#
#    
res={}
res['complexStates_new'] = complexStates

filetosave = 'runData\\res\\perf_AnonPcrit_complexst.pkl'

if not os.path.isfile(filetosave):
    with open(filetosave, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")