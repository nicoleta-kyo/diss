# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:32:05 2020

@author: Niki
"""

"""
to-do's:s
- how to handle goal state
- probabilities to make states unattainable


"""

import numpy as np 
from gen_states_functions import gen_costs, gen_combs, gen_base, gen_state_names, add_state, letttonum


#numSteps = 15

class LettersEnv():

  def __init__(self, states=None, num_states=None, max_cost=None, goal_state=None):
      
      self.defaultMaxComb = 10
      
      if states is None:
          self.states = self.gen_states(num_states, max_cost)
      else:
          self.states = states
          self.max_cost = self.states.cost.iloc[-1]
          
      self.t = 0
      self.num_act = 4
      self.num_combs = self.states.shape[0]    # number of combinations
      if goal_state is None:
          self.goal_state = self.num_combs-1   # default goal state is the one of highest cost
      
  def step(self, action):
      self.ssm[0,self.t] = action
      
      # look for state reached for each cost
      for cost in range(2,self.max_cost+1):
          if self.t+1-cost < 0:
              break
          self.ssm[cost-1, self.t] = self._return_state(cost)
      
      #retuning array is tricky - why not try returning max cost state achieved?
#      obs = self.ssm[1:, self.t]
      argst = self.ssm[:,self.t] > -1
      obs = self.ssm[argst,self.t][-1]
      
      reward = self._return_reward(obs)
      done = True if reward != 0 else False
    
      self.t += 1
      # return the array with the complex states reached/not reached
      return obs, reward, done
      
  def _return_state(self,cost):
      
      matched = -1      
      # only look for a state within the allowed number of time steps
      # since a previous state of the desired cost was reached
      if (np.unique(self.ssm[cost-1, self.t+1-cost:self.t]) == -1).all():
          # get the action sequence to be matched to a state of the desired cost
          act_seq = self.ssm[0, self.t+1-cost:self.t+1]  
          act_tomatch = ''.join([str(act_seq[i]) for i in range(len(act_seq))])
          # check if sequence matches with a state
          if not self.states[self.states.bitseqint == act_tomatch].empty:
              matched_str = self.states[self.states.bitseqint == act_tomatch].iloc[0,0]
              # output as observation index of the state + 1 (0 is empty start)
              matched = self.states[self.states.bitseqint == act_tomatch].index[0]
    
      return matched  

  def _return_reward(self, obs):
#      if self.goal_state in obs:
      if self.goal_state == obs:
          rew = 1
      else:
          rew = 0
      return rew
      
  def reset(self, numSteps):
      # store what the agent did as actions for the current episode
      # as well as track state progression
      self.ssm = np.repeat(-1, self.max_cost*numSteps).reshape(self.max_cost, numSteps)   # my state_sequence_memory
      self.t = 0
      
      #initial state is 0
      return -1
  
  def gen_states(self, num_states=None, max_cost=None, max_comb=None):
      
      if max_comb is None:
          max_comb = self.defaultMaxComb
      self.max_cost = max_cost
      states = gen_base()
      stns = gen_state_names()
      
      costs = gen_costs(num_states, max_cost)
      combs = gen_combs(num_states, max_comb)
          
      for comb,cost in zip(combs, costs):
          stn = np.setdiff1d(stns, states.state)[0]  # define the name of the new state
          add_state(states, stn, comb, cost)
          
      # add 0-4 state values
      states['bitseqint'] = letttonum(states.bitseq.values)
      
      return states
      
  def gen_ex_states(self):
      
      import pandas as pd
        
      states = gen_base()
      df2 = pd.DataFrame([['T1', 2, 'AC', 'AC'],
                            ['Q', 2, 'CB', 'CB'],
                            ['G', 3, 'T1B', 'ACB'],
                            ['F', 2, 'CC', 'CC'],
                            ['P', 2, 'CA', 'CA'],
                            ['Y', 5, 'GF', 'ACBCC'],
                            ['Q', 6, 'YA', 'ACBCCA']
                            ], columns = ['state', 'cost', 'comb', 'bitseq'])
      states=states.append(df2)
      states.reset_index(drop=True, inplace=True)
      # add int bitzeq
      states['bitseqint'] = letttonum(states.bitseq.values)
      self.max_cost = 6
      
      return states
  
  # return states traversed for the episode without the base states
  def return_traversed_states(self):
      return len(np.unique(self.ssm[self.ssm > 3]))
           
      
