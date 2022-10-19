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
from gen_states_functions_2 import gen_costs, gen_combs, gen_base, gen_state_names, add_state, letttonum



# Create states for runs

max_cost = 10
num_states = 10
stns = gen_state_names()

states = gen_base()

#!!! this is not great - very hard-coded
if max_cost == 10:
    cost_gap = 1
elif max_cost == 25:
    cost_gap = 2
else:
    cost_gap = 3
    
costs = gen_costs(num_states, max_cost, cost_gap)
combs = gen_combs(num_states) 
       
for comb,cost in zip(combs, costs):
  stn = np.setdiff1d(stns, states.index)[0]  # define the name of the new state
  states = add_state(states, stn, comb, cost)
          
# add 0-4 state values
states['bitseqint'] = letttonum(states.bitseq.values)
    
obj={}
obj['states'] = states

filenm = 'states\\states_cost10_newenv.pkl'

# save
if not os.path.isfile(filenm):
    with open(filenm, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
else:
    print("File already exists!")  
    

    