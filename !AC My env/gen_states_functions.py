# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:51:58 2020

@author: Niki
"""


import pandas as pd
import numpy as np
import string
import numpy.random as random



# generate n values between 0 and the limit (max cost) with minimum 3 between each value
def gen_costs(n, limit, gap):

    
    
#    mingap = 3     # min gap between the cost values
    slack = limit - gap * (n - 1)
    
    
    costs = np.array([0,1])
    steps = random.randint(0, slack)

    increments = np.hstack([np.ones((steps,)), np.zeros((n,))])
    np.random.shuffle(increments)

    locs = np.argwhere(increments == 0).flatten()
    costs = np.cumsum(increments)[locs] + gap * np.arange(0, n)
    costs[ costs == 0 ] = 2   # don't want costs = 0 or 1, so if there are such, make them 2
    costs[ costs == 1 ] = 2
    costs[-1] = limit  # make state of the limit cost
       
    return costs.astype(int)

# generate random n combination values for each state
def gen_combs(n, max_combs):
    
    num_uniques = int(n/4)
    
    combs = np.random.randint(2,max_combs,n)
    combs[np.random.randint(0,n,num_uniques)] = 1  # randomly force some to have only one comb

    return combs

# Create base states
def gen_base():
    st = [['A', 1], ['B', 1], ['C', 1], ['D', 1]]
    states = pd.DataFrame(st, columns = ['state', 'cost'])
    states['comb'] = ['A', 'B', 'C', 'D']
    states['bitseq'] = states['comb']
    
    return states

    # make list of state names to use 
def gen_state_names():
    lett = [i for i in string.ascii_uppercase]
    letters=[]
    while len(letters) < 26*100:
        letters+=lett
    nums = np.repeat(np.arange(100), 26)
    state_names = lett
    state_names += [str(l_)+str(n_) for l_, n_ in zip(letters,nums)]
    
    return state_names


# Add a state to the states df
def add_state(states, stname, num_combs, cost):
    
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


# convert letters to numbers
def letttonum(abc_array):
    newarr = [None] * len(abc_array)
    for i in range(len(abc_array)):
        abc = abc_array[i]
        new = ''
        for char in abc:
            if char == 'A':
                new+='0'
            elif char == 'B':
                new+='1'
            elif char == 'C':
                new+='2'
            elif char == 'D':
                new+='3'
        newarr[i] = new
    
    return newarr

