# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:16:29 2020

@author: Niki
"""




import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import time
from lettersEnv import LettersEnv

import pdb


class ising:
    # Initialize the network
    def __init__(self, netsize, Nsensors=1, Nmotors=1, predictor=None):  # Create ising model

#         pdb.set_trace()
        
        self.size = netsize		#Network size
        self.Inpsize = Nsensors  # Number of sensors
        self.Msize = Nmotors  # Number of motors

        self.h = np.zeros(netsize) # local biases
        self.J = np.random.normal(0, 0.1, (self.size, self.size)) # symmetic weights between hidden variables
        
        self.defaultGamma = 0.999  # as in Otsuka - Solving POMPDs
        self.defaultLr = 0.01  # as in Otsuka - Solving POMPDs
        
        self.defaultBeta = 1
    
    def connectToEnv(self, states=None, num_states=None, max_cost=None, goal_state=None):
        if states is not None:
            self.env = LettersEnv(states=states)
        else:
            self.env = LettersEnv(num_states, max_cost, goal_state)
        self.maxobs = self.env.num_combs
        self.numact = self.env.num_act

    # Transorm the sensor/motor input into integer index
    def InputToIndex(self, x, xmax, bitsize):
        return int(np.floor((x + xmax) / (2 * xmax + 10 * np.finfo(float).eps) * 2**bitsize))
           
    # Calculate hat_h_k for each k (expert)    
    def ExpectedValueExperts(self, sensors, motors):
        # self.s[Ssize:-Msize] - all hidden
        # self.h[Ssize:-Msize] - biases of hidden
        # self.J[:Ssize,Ssize:-Msize] - sensor-hidden connections
        # self.J[Ssize:-Msize, -Msize:] - hidden-motor conn.
        
        num_e = self.size - (self.Inpsize+self.Msize)  # number of hidden units (experts)
        ve = np.zeros(num_e)             # array of expected values for experts
        for k in range(num_e):
            ve[k] = sigmoid(np.dot(self.J[0:self.Inpsize,self.Inpsize:-self.Msize][:,k], sensors) + 
                            np.dot(self.J[self.Inpsize:-self.Msize, -self.Msize:][k,:], motors) +
                            self.h[self.Inpsize:-self.Msize][k])
        
        return ve
    
    # takes in state and action and uses their binary equivalents to calculate the energy (not the actual network's)
    # state_bit = binarised observation + memory
    # action = the int action
    def CalcFreeEnergy(self, state_bit, action):
        
#         pdb.set_trace()
          
        #binarise action
        action_bit = action2bit(action, self.numact)
    
        ve = self.ExpectedValueExperts(state_bit, action_bit).reshape(-1,1)  #calculate expected hidden values
        ss = state_bit.reshape(1,-1)
        sm = action_bit.reshape(1,-1) 
        
        # calculate the negative log-likelihood
        a = -np.dot(np.dot(ss, self.J[:self.Inpsize,self.Inpsize:-self.Msize]), ve)    # 1
        b = -np.dot(ss, self.h[:self.Inpsize].reshape(-1,1))                             # 2
        c = -np.dot(np.dot(sm, np.transpose(self.J[self.Inpsize:-self.Msize, -self.Msize:])), ve)  # 3
        d = -np.dot(sm, self.h[-self.Msize:].reshape(-1,1))                            # 4
        e = -np.dot(self.h[self.Inpsize:-self.Msize], ve)                       # 5
        
        # calculate the negative entropy
        f = 0                                                           # 6 + 7
        for k in range(len(ve)):
            f += ve[k]*np.log(ve[k] ) + (1 - ve[k])*np.log(1 - ve[k])
        
        return round(float(a+b+c+d+e+f),2)
    
    def SarsaLearning(self, total_episodes, max_steps, Beta=None, lr=None, gamma=None):
        
        self.total_episodes = total_episodes
        self.log = np.tile(np.repeat(float('inf'),5),(total_episodes, 1))  # track learning
        
        for episode in range(total_episodes):
            
            tic = time.perf_counter()
            
            self.beta = Beta[episode] if Beta is not None else self.defaultBeta
            
            # reset the environment
            state = self.env.reset(max_steps)
            state_bit = bitfield(self.InputToIndex(state, self.maxobs, self.Inpsize), self.Inpsize) # binarise observation

            action = self.ChooseAction(state_bit)
           
            # calculate Q(s,a)
            Q1 = -1*self.CalcFreeEnergy(state_bit, action)

            self.initialiseDeltas()  # update of senhid weights, hidmot weights and the biases for the sen, mot and hid
            Qs = np.repeat(-np.inf, max_steps)
            
            t = 0
            while t < max_steps:             
                
                # step with a and receive obs'
                state2, reward, done  = self.env.step(action)
                state2_bit = bitfield(self.InputToIndex(state2, self.maxobs, self.Inpsize), self.Inpsize) # binarise observation
                
                action2 = self.ChooseAction(state2_bit)
                
                # calculate Q(s',a')
                Q2 = -1*self.CalcFreeEnergy(state2_bit, action2) # calculate Q2 = Q(s',a') = -F(s',a')
                
                
                # calculate update to the weights of the network currently having s and a
                self.SarsaUpdateOnline(Q1, reward, Q2, state_bit, action, gamma)
                
                Qs[t] = Q1

                # updates for loop
                state = state2  # update obs = obs'   (needed to calculate m' = obs + a)
                state_bit = state2_bit
                action = action2  # update a = a'  (needed to calculate m' = obs + a and to also get obs')
                Q1 = Q2  # no need to update m (memory), because Q1 is directly updated

                t += 1
                
                if done:
                    break
                
            # apply episodic update rule: t+1 is the number of time steps out of the max actually performed
            self.SarsaUpdateEpisodic(t)
            
            # update log
            vishidJ = np.vstack((self.J[:self.Inpsize,self.Inpsize:-self.Msize], np.transpose(self.J[self.Inpsize:-self.Msize, -self.Msize:])))
            self.log[episode, :] = np.array([state, reward, np.mean(Qs[:t]), np.mean(np.abs(vishidJ)), np.max(np.abs(vishidJ))])
            
            toc = time.perf_counter()
#            print('Episode ' + str(episode) + ' took ' + str(int((toc - tic)/60)) + ' minutes.')

    def SarsaUpdateOnline(self, Q1, reward, Q2, state_bit, action, gamma):
        
#         pdb.set_trace()
        
        if gamma is None:
            gamma = self.defaultGamma
        
        # TD error
#         rDiff = lr*(reward + gamma * Q2 - Q1)
        rDiff = reward + gamma * Q2 - Q1
        
        action_bit = action2bit(action, self.numact)
        # calculate dQ/Wsensors
        ve = self.ExpectedValueExperts(state_bit, action_bit).reshape(1,-1)
        ve_new = np.repeat(ve, len(state_bit)).reshape(-1,1)
        s_ne = np.repeat(state_bit.reshape(1,-1), ve.shape[1], axis=0)
        s_new = s_ne.reshape(s_ne.shape[0]*s_ne.shape[1],1)
        dQdWs = np.multiply(ve_new,s_new)
        
        # calculate dQ/Wmotors
        ve_new = np.repeat(ve, len(action_bit)).reshape(-1,1)
        m_ne = np.repeat(action_bit.reshape(1,-1), ve.shape[1], axis=0)
        m_new = m_ne.reshape(m_ne.shape[0]*m_ne.shape[1],1)
        dQdWm = np.multiply(ve_new,m_new)
        
        # updates weights of sensors and motors
        num_h = self.size - (self.Inpsize+self.Msize)
        self.dWs += (rDiff*dQdWs).reshape(num_h, self.Inpsize)
        self.dWm += (rDiff*dQdWm).reshape(num_h, self.Msize)
        
        # updtes biases of sensors, motors, hidden
        self.dhs += rDiff*state_bit
        self.dhm += rDiff*action_bit
        self.dhh += rDiff*(ve.reshape(-1))
    
    def SarsaUpdateEpisodic(self, T, lr=None):
        
        if lr is None:
            lr = self.defaultLr
        
        # update weights
        self.J[:self.Inpsize,self.Inpsize:-self.Msize] += lr*(self.dWs/T).T
        self.J[self.Inpsize:-self.Msize, -self.Msize:] += lr*(self.dWm/T)
        
        # update biases
        self.h[:self.Inpsize] += lr*(self.dhs/T)
        self.h[-self.Msize:] += lr*(self.dhm/T)
        self.h[self.Inpsize:-self.Msize] += lr*(self.dhh/T)
        
    
    def initialiseDeltas(self):
        num_h = self.size - (self.Inpsize+self.Msize)
        
        self.dWs = np.zeros((num_h, self.Inpsize))
        self.dWm = np.zeros((num_h, self.Msize))
        self.dhs = np.zeros(self.Inpsize)
        self.dhm = np.zeros(self.Msize)
        self.dhh = np.zeros(num_h)
    
    # state = observation + memory
    def ChooseAction(self, state):
        
#         pdb.set_trace()

        # calculate probabilities of all actions - based on Otsuka's paper
        p_a = np.zeros((self.env.num_act,2))
        for a in range(self.env.num_act):
            if -self.beta*self.CalcFreeEnergy(state, a) > 700:
                bfe = 700
                print("-beta*FE reaching maximum for exponential in chooseAction method, therefore bounding it to 700.")
            else:
                bfe = -self.beta*self.CalcFreeEnergy(state, a)
            fa = np.exp(bfe)
            fa_ = 0
            for a_ in range(self.env.num_act):
                if -self.beta*self.CalcFreeEnergy(state, a_) > 700:
                    bfe = 700 
                    print("-beta*FE reaching maximum for exponential in chooseAction method, therefore bounding it to 700.")
                else:
                    bfe = -self.beta*self.CalcFreeEnergy(state, a_)
                fa_ = fa_ + np.exp(bfe)
            p_a[a,1] = fa/fa_
            p_a[a,0] = a  #add index column

        # sample an action from the distribution
        ord_p_a = p_a[p_a[:,1].argsort()]
        ord_p_a[:,1] = np.cumsum(ord_p_a[:,1])

        b = np.array(ord_p_a[:,1] > np.random.rand())
        act = int(ord_p_a[b.argmax(),0])  # take the index of the chosen action
 
        return act
    
    # performane for sole rbm is total rewards/ episodes
    def getPerf(self):
        return np.sum(self.log[:,1])/self.total_episodes
    
    def displayRunData(self, display=True, save=False, savePath=None, plotname=None):
        
        res = self.log
        x = range(self.total_episodes)

        # reward
    
        y = res[:,1]
        l1 = 'ext_reward'
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label=l1)
        ax.legend()
        plt.title(plotname)
        
        fig1 = plt.gcf()
        
        if display:
            plt.show()
    
        # actor -free energy
    
        y = res[:,2]
        l2 = 'mean_q'
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label=l2)
        ax.legend()
        plt.title(plotname)
        
        fig2 = plt.gcf()
        if display:
            plt.show()
    
        # mean J
    
        y = res[:,3]
        l3='mean_j'
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label=l3)
        ax.legend()
        plt.title(plotname)
        
        fig3 = plt.gcf()
        if display:
            plt.show()
    
        # max J
    
        y = res[:,4]
        l4='max_j'
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label=l4)
        ax.legend()
        plt.title(plotname)
        
        fig4 = plt.gcf()
        if display:
            plt.show()
        
        if save==True:
            fig1.savefig(savePath+l1)
            fig2.savefig(savePath+l2)
            fig3.savefig(savePath+l3)
            fig4.savefig(savePath+l4)
            
        plt.close('all')

        
        
# Transform bool array into positive integer
# [nk] encodes the state of the neurons as one number
def bool2int(x):
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2**i
    return int(y)

# Transform positive integer into bit array
def bitfield(n, size):
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# actions to be converted to a discrete multinomial variable
def action2bit(action, max_act):
    bit_action = np.zeros(max_act)
    bit_action[action] = 1
    
    return bit_action

