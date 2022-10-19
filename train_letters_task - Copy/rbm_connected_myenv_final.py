# -*- coding: utf-8 -*-
"""

- Better code for calculations
- Adding regularisation


"""




import numpy as np
from itertools import product as prod
import matplotlib.pyplot as plt
import time
from lettersEnv_2 import LettersEnv
import warnings
import pdb


class ising:
    # Initialize the network
    def __init__(self, netsize, Nmemory, Nsensors=1, Nmotors=1, predictor=None):  # Create ising model

#         pdb.set_trace()
        
        self.size = netsize		#Network size
        self.Ssize = Nsensors
        self.Inpsize = Nsensors+Nmemory  # Number of sensors
        self.Msize = Nmotors  # Number of motors
        self.Hsize = netsize-Nsensors-Nmotors
        self.h = np.zeros((self.Inpsize+self.Msize+self.Hsize,1)) # local biases
        self.hidsen = np.random.normal(0, 0.1, (self.Hsize, self.Inpsize))
        self.hidmot = np.random.normal(0, 0.1, (self.Hsize, self.Msize))
        
        if predictor is not None:
            self.predictor = predictor
        
        self.defaultGamma = 0.999  # as in Otsuka - Solving POMPDs
        self.defaultLr = 0.01  # as in Otsuka - Solving POMPDs
        self.defaultLamb = 0.5 # L-2 regularisation
        
        self.defaultBeta = 1
    
    def connectToEnv(self, states=None, num_states=None, max_cost=None, goal_state=None):
        if states is not None:
            self.env = LettersEnv(states=states)
        else:
            self.env = LettersEnv(num_states, max_cost, goal_state)
        self.maxobs = self.env.num_combs
        self.numact = self.env.num_act
        
    # Calculate hat_h_k for each k (expert)    
    def ExpectedValueExperts(self, state_bit, action_bit):

        w = []
        with warnings.catch_warnings(record=True) as w:
            ve = sigmoid(np.dot(self.hidsen,state_bit) + np.dot(self.hidmot,action_bit) + self.h[self.Inpsize:-self.Msize])
            
        if not not w:
            pdb.set_trace()
        return ve
    
    # takes in state and action and uses their binary equivalents to calculate the energy (not the actual network's)
    # state_bit = binarised observation + memory
    # action = the int action
    def CalcFreeEnergy(self, state_bit, action):
        
        w=[]
        with warnings.catch_warnings(record=True) as w:
              
            #binarise state and action
#            state_bit = input2bit(state, self.maxobs, self.Inpsize)
            action_bit = input2bit(action, self.numact, self.Msize)
            ve = self.ExpectedValueExperts(state_bit, action_bit)  #calculate expected hidden values
            
           # calculate the energy of the state-action configuration (the negative log-likelihood)
            negloglike =  float(-np.dot(np.dot(self.hidsen, state_bit).T, ve)    # 1
                             -np.dot(self.h[:self.Inpsize].T, state_bit)     # 2
                             -np.dot(np.dot(self.hidmot, action_bit).T, ve)  # 3
                             -np.dot(self.h[-self.Msize:].T, action_bit)       # 4
                             -np.dot(self.h[self.Inpsize:-self.Msize].T, ve)    # 5
                            )
            # calculate the negative entropy
            negent = 0                                                           # 6 + 7
            for k in range(len(ve)):
                negent += float(ve[k]*np.log(ve[k] ) + (1 - ve[k])*np.log(1 - ve[k]))
        
        if not not w:
            pdb.set_trace()
        
        return (negloglike, negent, negloglike+negent)
    
    
    def SarsaLearning(self, total_episodes, max_steps, Beta=None, lr=None, gamma=None, lamb=None):
        
        self.total_episodes = total_episodes
        # store expected energy, negative entropy and free energy for each state + how many times the state was reached
        self.FEtable = np.zeros((total_episodes,self.env.num_combs,(3+1)))
        # store predictor's quality, internal reward, external reward, rewardUsedForLearning
        self.log = np.tile(np.repeat(-np.inf,4), (total_episodes, max_steps, 1) )
        # create history array 
        self.predictor.setHistory(total_episodes, max_steps)  
        

        for episode in range(total_episodes):
            
            
#            if episode % 1000 == 0 and episode != 0:
                
#                toc=time.perf_counter()
#                print('Next 1000 episodes up until ' + str(episode) + ' took ' + str(int((toc - tic)/60)) + ' minutes.')
#            
#            tic = time.perf_counter()
            
            self.beta = Beta[episode] if Beta is not None else self.defaultBeta
            self.lamb = lamb if lamb is not None else self.defaultLamb
            
            #memory is not reset per episode
            if episode == 0:
                memory = self.predictor.laststate
            else:
                memory = memory2
            
            # reset the environment
            state = self.env.reset(max_steps)
            
            # choose a (action) based on s = obs+m (obs+memory)
            state_memory = np.vstack([input2bit(state, self.maxobs, self.Ssize),memory.reshape(-1,1)])
            action = self.ChooseAction(state_memory)
           
            # calculate Q(s,a)
            fe1 = self.CalcFreeEnergy(state_memory, action)
            Q1 = -1*fe1[2]

            self.initialiseDeltas()  # update of senhid weights, hidmot weights and the biases for the sen, mot and hid            
            t = 0
            while t < max_steps:             
                
                # calculate m'
                esn_input = np.array([state, action]).reshape(1,-1)
                memory2 = self.predictor.get_states(esn_input, extended=False, continuation=True) # we only take state activations, not concatenate states with input which is done to train the weights and to also predict                                                                           
                
                # step with a and receive obs'
                state2, ext_reward, done  = self.env.step(action)    
                
                #--
                
                # Predictor improvement and calculation of internal reward
                self.predictor.history[episode,t,:] = np.array([state, action, state2, ext_reward])  # update predictor history
                int_reward, qualp = self.predictor.calculateInternalReward() # calculate internal reward
            
                # Calculation of reward for SARSA update
                reward = int_reward+ext_reward
                
                #--         
                
                # choose a' based on s' = obs'+m'
                # !!! mind dimensions
                state_memory2 = np.vstack([input2bit(state2, self.maxobs, self.Ssize),memory2.reshape(-1,1)])  # add the predictor's state to the obsrvations
                action2 = self.ChooseAction(state_memory2)
                
                # calculate Q(s',a')
                fe2 = self.CalcFreeEnergy(state_memory2, action2)
                Q2 = -1*fe2[2] # calculate Q2 = Q(s',a') = -F(s',a')           
                
                # calculate update to the weights of the network currently having s and a
                self.SarsaUpdateOnline(Q1, reward, Q2, state_memory, action, gamma)
                
                #store fe2 for state2
                self.FEtable[episode,state2,1:] += fe2
                self.FEtable[episode,state2,0] += 1  #mark that state was reached

                # update log
                self.log[episode, t, :] = [qualp, int_reward, ext_reward, reward]
                
                # updates for loop
                state_memory = state_memory2
                state = state2
                action = action2  # update a = a'  (needed to calculate m' = obs + a and to also get obs')
                fe1=fe2
                Q1 = Q2  # no need to update m (memory), because Q1 is directly updated

                t += 1
                
                if done:
                    break
                
            # apply episodic update rule: t+1 is the number of time steps out of the max actually performed
            self.SarsaUpdateEpisodic(T=1,lamb=lamb)
            # calculate mean free energies
            for i in range(self.FEtable.shape[1]):
                if self.FEtable[episode,i,0] > 0:
                    self.FEtable[episode,i,1:] = self.FEtable[episode,i,1:]/self.FEtable[episode,i,0]

            


    def SarsaUpdateOnline(self, Q1, reward, Q2, state_bit, action, gamma):
        
#         pdb.set_trace()
        
        if gamma is None:
            gamma = self.defaultGamma
        
        # TD error
        rDiff = reward + gamma * Q2 - Q1
        
        # values experts
#        state_bit = input2bit(state, self.maxobs, self.Inpsize)
        action_bit = input2bit(action, self.numact, self.Msize)
        ve = self.ExpectedValueExperts(state_bit, action_bit)
        
        # calculate dQ/Wsensors and dQ/Wmotors
        dQdWs = np.array([x*y for x,y in prod(ve,state_bit)]).reshape(self.Hsize,self.Inpsize)
        dQdWm = np.array([x*y for x,y in prod(ve,action_bit)]).reshape(self.Hsize,self.Msize)
        
        # updates weights of sensors and motors
        self.dWs += rDiff*dQdWs
        self.dWm += rDiff*dQdWm
        # updtes biases of sensors, motors, hidden
        self.dh += rDiff*np.vstack([state_bit, ve, action_bit])
    
    
    def SarsaUpdateEpisodic(self, T, lr=None, lamb=None):
        
        if lr is None:
            lr = self.defaultLr
            
        if lamb is None:
            lamb = self.lamb
        
        # update weights
        self.hidsen += lr*(self.dWs/T - lamb*self.hidsen)
        self.hidmot += lr*(self.dWm/T - lamb*self.hidmot)
        # update biases
        self.h += lr*(self.dh/T - lamb*self.h)
        
        
#                    
#    def limitGrowth(self,params):
#        
#        # restrict them from growing too much
#        for i in range(params.shape[0]):
#            for j in range(params.shape[1]):
#                if abs(params[i,j]) > self.maxWeight:
#                    params[i,j] = self.maxWeight*np.sign(params[i,j])

    
    def initialiseDeltas(self):
        
        self.dWs = np.zeros(self.hidsen.shape)
        self.dWm = np.zeros(self.hidmot.shape)
        self.dh = np.zeros(self.h.shape)
    
    # state = observation + memory
    def ChooseAction(self, state):
        
        w = []
        with warnings.catch_warnings(record=True) as w:
            # calculate probabilities of all actions - based on Otsuka's paper
            p_a = np.zeros((self.env.num_act,2))
            for a in range(self.env.num_act):
                FE = self.CalcFreeEnergy(state, a)[2]
                fa = np.exp(-self.beta*FE)
                fa_ = 0
                for a_ in range(self.env.num_act):
                    FE = self.CalcFreeEnergy(state, a_)[2]
                    fa_ = fa_ + np.exp(-self.beta*FE)
                p_a[a,1] = fa/fa_
                p_a[a,0] = a  #add index column
    
            # sample an action from the distribution
            ord_p_a = p_a[p_a[:,1].argsort()]
            ord_p_a[:,1] = np.cumsum(ord_p_a[:,1])
    
            b = np.array(ord_p_a[:,1] > np.random.rand())
            act = int(ord_p_a[b.argmax(),0])  # take the index of the chosen action
            
        if not not w:
            pdb.set_trace()
        
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

        
  
# sigmoid for array of shape (,x) or (1,x)
def sigmoid(x):
    
    if x.ndim < 2:
        x=x.reshape(-1,1)
        
    y=np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        y[i] = 1 / (1 + np.exp(-x[i]))
        
    return y
      
# Transform bool array into positive integer
# [nk] encodes the state of the neurons as one number
def bool2int(x):
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2**i
    return int(y)


# Transform env input to bit array
def input2bit(x, xmax, bitsize):
    ind = int(np.floor((x + xmax) / (2 * xmax + 10 * np.finfo(float).eps) * 2**bitsize))
    x = [int(x) for x in bin(int(ind))[2:]]
    x = [0] * (bitsize - len(x)) + x
    return np.array(x).reshape(len(x),1)


