# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:51:58 2020

@author: Niki
"""


import pandas as pd
import numpy as np
import string
import itertools as its
import numpy.random as random


########## Generate desired costs and number of combinations for creation of states

n = 20
limit = 100
mingap = 3

slack = limit - mingap * (n - 1)

def generate():
    steps = random.randint(0, slack)

    increments = np.hstack([np.ones((steps,)), np.zeros((n,))])
    np.random.shuffle(increments)

    locs = np.argwhere(increments == 0).flatten()
    return np.cumsum(increments)[locs] + mingap * np.arange(0, n)

costs = generate()

num_combs = np.random.randint(1,10,n)



#============ Function to add new states to  DF states
def make_state(num_combs, cost):
    stname = np.setdiff1d(state_names, states.state)[0]  # define the name of the new state
    comb=0
    while comb < num_combs:  # need to add a stop criterion if all combs exhausted
        choice_st_bitseq = pd.DataFrame(columns = ['choice_state', 'choice_bitseq'])
        new_cost = 0    
        #concatenate existing states until desired cost
        while new_cost < cost:
            if comb==0 and choice_st_bitseq.empty: # get the biggest state with cost < wanted cost
                ord_costs=np.unique(states.sort_values(by='cost').cost.values)
                m=1
                while ord_costs[-m] >= cost:
                    m+=1
                choice_state = np.random.choice(states[states.cost == ord_costs[-m]].state)
            else:
                choice_state = np.random.choice(states.state)
            choice_bitseq = np.random.choice(states[states.state == choice_state].bitseq)
            c = new_cost + states[states.state == choice_state].cost.iloc[0]
            # add new state to the combination array
            if c < cost or (c == cost and not choice_st_bitseq.empty):
                new_cost = c
                choice_st_bitseq = choice_st_bitseq.append(pd.DataFrame(np.array([choice_state, choice_bitseq]).reshape(1,2), columns = choice_st_bitseq.columns))
        choice_st_bitseq = choice_st_bitseq.sample(frac=1) #shuffle and construct new bitseq
        new_comb = ''.join(choice_st_bitseq.choice_state.values)
        new_bitseq = ''.join(choice_st_bitseq.choice_bitseq.values)
        # check if new comb is unique
        if new_bitseq not in states.comb.values:
            states.loc[len(states)] = [stname, cost, new_comb, new_bitseq] 
            comb += 1
            
            


#========== Create starting sttaes and DF

st = [['A', 1], ['B', 1], ['C', 1], ['D', 1]]
states = pd.DataFrame(st, columns = ['state', 'cost'])
states['comb'] = ['A', 'B', 'C', 'D']
states['bitseq'] = states['comb']


# make list of state names to use 
lett = [i for i in string.ascii_uppercase]
letters=[]
while len(letters) < 26*100:
    letters+=lett
nums = np.repeat(np.arange(100), 26)
state_names = lett
state_names += [str(l_)+str(n_) for l_, n_ in zip(letters,nums)]




#=========== Create
for comb,cost in zip(num_combs, costs):
    make_state(comb,cost)

#=========== Save object
     
        
        