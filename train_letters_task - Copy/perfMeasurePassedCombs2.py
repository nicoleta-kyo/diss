# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:07:11 2020

@author: Niki
"""
import pandas as pd
import numpy as np


def getStateNums(states):
    
    # num bitseqs
    bitnum=states.groupby(level=0).count().cost
    # num combs
    combnum=states.groupby(level=0).comb.nunique()
    # combine
    dic = {}
    dic['bitnum'] = bitnum
    dic['combnum'] = combnum
    
    statenum = pd.DataFrame(dic)
    
    return statenum

    
def checkNumbitsBeforeN(hist, states, stateN):
    
    statenum = getStateNums(states)
    
    #take the times the state happened
    eps,steps=np.where(hist[:,:,2] == stateN)
    costStateN = states.loc[stateN].cost.iloc[0]
    
    # for each one collect the comb by looking at the bitseqint
    combStat = {}
    combNum = {}  # tuple episodes, times
    combFin= {}
    

    lastEp=0
    reached=[]
    for ep,step in zip(eps,steps):
#        print(str(count) + ' out of ' + str(len(eps)))
        reached = [] if lastEp != ep else reached #keep track of different combinations and only do the analysis for the first time it appears per episode
        start = step-costStateN+1
        stop = step + 1
        seq = ''.join(map(str,[*hist[ep,start:stop,1]]))
        if len(seq) != 0:
            # get an underlying complex state and see how many different ways it was reached
            stateBeforeN = []
            stateBeforeN = [i for i in states[states.bitseqint == seq].comb.values[0] if i not in ['A','B','C','D']] 
            if not not stateBeforeN:
                for i,val in enumerate(stateBeforeN):
                    stateBefore = np.unique(states[states.state == val].index)[0]  # get the index of the state
                    if stateBefore not in reached:
                        reached.append(stateBefore)
                        
                        costStateBefore = np.unique(states.loc[stateBefore].cost)[0]
                        acts = hist[ep, :step, 1] # take history in episode before n happened
                        occs=getOccState(states, acts,stateBefore) # identify when n-1 happened
                        # collect the bitseqs of n-1 that were actioned
                        seqs = []
                        for iocc,vocc in enumerate(occs):
                            start = vocc-costStateBefore+1
                            stop = vocc + 1
                            seqs.append( ''.join(map(str,[*acts[start:stop]])) )
                        num=len(np.unique(seqs))  # number of bitseqs reached    
                       
                        # store
                        if num != 0 :
                            combStat[stateBefore] = combStat[stateBefore] + num if stateBefore in combStat.keys() else num
                            combNum[stateBefore] = combNum[stateBefore] + 1 if stateBefore in combNum.keys() else 1
        lastEp=ep
    
    for i in combStat.keys():
        tot=statenum.bitnum.loc[i] # total number of bitseqs
        combFin[i] = round((combStat[i]/combNum[i])/tot,2) # 
    
    return combFin

    
#--
def getOccState(states, seqActs, state):
    cost = np.unique(states.cost.loc[state])[0]
    numSteps=len(seqActs)
    ssm = np.repeat(-1, 2*numSteps).reshape(2,numSteps)
    
    for actind,actval in enumerate(seqActs):
      ssm[0,actind] = actval
      if actind+1-cost >= 0:
          matched = -1      
          # only look for a state within the allowed number of time steps
          # since a previous state of the desired cost was reached
          if (np.unique(ssm[1, actind+1-cost:actind]) == -1).all():
              # get the action sequence to be matched to a state of the desired cost
              act_seq = ssm[0, actind+1-cost:actind+1]  
              act_tomatch = ''.join([str(act_seq[i]) for i in range(len(act_seq))])
              # check if sequence matches with a state
              if not states[states.bitseqint == act_tomatch].empty:
                  # output as observation index of the state + 1 (0 is empty start)
                  matched = states[states.bitseqint == act_tomatch].index[0]
                  ssm[1, actind] = matched
    a,=np.where(ssm[1] == state)
    
    return a
#
                      
