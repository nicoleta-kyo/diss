# -*- coding: utf-8 -*-
"""
Generate states sets

states10 - max cost 10
states25 - max cost 25
states50 - max cost 50

"""

import numpy as np
import pickle
import os
from gen_states_functions import gen_costs, gen_combs, gen_base, gen_state_names, add_state, letttonum



# Create states for runs


# 1to iterate!!
max_costs = [10,25,50]      #maximum cost a state can have
goal_hierch = [True,False]

#these remain the same but will still save them as params
num_states = 10   #actual big states - doesnt really matter for ext reward so can be kept like this
max_comb = 10
stns = gen_state_names()

#
def create_states(max_cost, goal_hierch):
    states = gen_base()
    
    #!!! this is not great - very hard-coden
    if max_cost == 10:
        cost_gap = 1
    elif max_cost == 25:
        cost_gap = 2
    else:
        cost_gap = 3
        
    costs = gen_costs(num_states, max_cost, cost_gap)
    combs = gen_combs(num_states, max_comb)
    combs[-1] = 1 if goal_hierch else 2   #make highest cost state to have only one comb which will contain last biggest cost: might be of use to the network
    
           
    for comb,cost in zip(combs, costs):
      stn = np.setdiff1d(stns, states.state)[0]  # define the name of the new state
      add_state(states, stn, comb, cost)
              
    # add 0-4 state values
    states['bitseqint'] = letttonum(states.bitseq.values)
    
    #save
    
    g = '_hierch' if goal_hierch else '_nonhierch'
    filenm = 'states//states_max_cost' + str(max_cost) + g +'.pkl'
    
    obj = {}
    obj['num_states'] = num_states
    obj['max_comb'] = max_comb
    obj['max_cost'] = max_cost
    obj['states'] = states
    obj['goal_hierch'] = goal_hierch
    
    if not os.path.isfile(filenm):
        with open(filenm, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("File already exists!")

for ic, c in enumerate(max_costs):
    for ig, g in enumerate(goal_hierch):
        create_states(c,g)
    
  
## load
#        
#file = 'states\\states_max_cost10_hierch.pkl'
#
#with open(file, 'rb') as inp:
#    std = pickle.load(inp)   
#
#states = std['states']    
        
        