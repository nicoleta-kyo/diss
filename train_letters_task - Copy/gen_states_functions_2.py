# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:54:09 2020

@author: Niki
"""



import pandas as pd
import numpy as np
import string
import numpy.random as random
import itertools


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
def gen_combs(n):
    
    num_uniques = int(n/4)
    
    combs = np.random.randint(2,4,n)
    combs[np.random.randint(0,n,num_uniques)] = 1  # randomly force some to have only one comb

    return combs

# Create base states
def gen_base():
    st = [['A', 1], ['B', 1], ['C', 1], ['D', 1]]
    states = pd.DataFrame(st, columns = ['state', 'cost'])
    states['comb'] = ['A', 'B', 'C', 'D']
    states['bitseq'] = states['comb']
    states.set_index('state',inplace=True)
    
    return states

    # make list of state names to use 
def gen_state_names(short=True):
    
    if not short:
        lett = [i for i in string.ascii_uppercase]
        letters=[]
        while len(letters) < 26*100:
            letters+=lett
        nums = np.repeat(np.arange(100), 26)
        state_names = lett
        state_names += [str(l_)+str(n_) for l_, n_ in zip(letters,nums)]
    else:
        state_names = [i for i in string.ascii_uppercase]
    
    return state_names



# Add a state to the states df
def add_state(states, stname, num_combs, cost):
    
   comb=0
   while comb < num_combs:  # need to add a stop criterion if all combs exhausted
      choices_st = []
      choices_bitseq = {}
      new_cost = 0    
      
      #concatenate existing states until desired cost
      while new_cost < cost:
          if comb==0 and not choices_st: # get the biggest state with cost < wanted cost
              ord_costs=np.unique(states.sort_values(by='cost').cost.values)
              m=1
              while ord_costs[-m] >= cost:
                  m+=1
              choice_state = np.random.choice(np.unique(states[states.cost == ord_costs[-m]].index))
          else:
              choice_state = np.random.choice(np.unique(states.index))
          choice_bitseqs = states[states.index == choice_state].bitseq.values.tolist()   
          
#          choice_bitseq = np.random.choice(states[states.state == choice_state].bitseq)
          c = new_cost + states[states.index == choice_state].cost.iloc[0]
          # add new state to the combination
          if c < cost or (c == cost and choices_st):
              new_cost = c
              # save the bitseqs of the chosen state
              choices_bitseq[choice_state] = choice_bitseqs
              # save the chosen state to the df to shuffle
              choices_st.append(choice_state)
              
      random.shuffle(choices_st) #shuffle and construct new bitseq
      new_comb = ''.join(choices_st)
      
      # check uniqueness of constructed combination and add to the states df as rows of all its bitseqs 
      if new_comb not in states.comb.values:
         
          # get the bitseqs of each comb
          bitlists = []
          for s in range(len(choices_st)):
              key=choices_st[s]
              b = choices_bitseq[key]
              bitlists.append(b)
          # create all new bitseqs between the separate bitseqs of the keys
          new_bitseqs = []
          for i in itertools.product(*bitlists):
              new_bitseqs.append(''.join(i))
          # construct the new rows to add to the states
          toAdd = pd.DataFrame(columns = ['state', 'cost', 'comb', 'bitseq'])
          for i,new_bitseq in enumerate(new_bitseqs):
              toAdd.loc[i] = [stname, cost, new_comb, new_bitseq]
          toAdd.set_index('state',inplace=True)
          # add them
          states = states.append(toAdd)                           
          comb += 1

   return states

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

