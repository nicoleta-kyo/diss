"""
Code for ising taken from Aguilera's code for "Adaptation to Criticality":
Changes:
- Restricted - connections within hidden layer not being updated
- Embodiment for environment Frozen Lake
- RL Learning by a variant of SARSA - to Hinton's paper "Using Q-energies"
     *using sensors and motors variables (not the state variable)
     
For Otsuka's model architecture:
- Memory layer
- Predictor ESN as parameter so it can have access to its reservoir and history
- Sensors are 0/1 for SARSA learning
- SARSA learning includes updating the predictor's history and getting its internal reward
internal reward is taken only of non-negative and is bounded by tanh function
The rule update is episodic!

""" 


import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import gym

import pdb


class ising:
    # Initialize the network
    def __init__(self, netsize, Nmemory, Nsensors=1, Nmotors=1, predictor=None):  # Create ising model

#         pdb.set_trace()
        
        self.size = netsize		#Network size
        self.Ssize = Nsensors  # Number of sensors
        self.Msize = Nmotors  # Number of motors
        self.Memsize = Nmemory # Number of memory units
        self.Inpsize = Nsensors + Nmemory  # Number of sensors+memory

        self.h = np.zeros(netsize) # local biases
        self.J = np.random.normal(0, 0.1, (self.size, self.size)) # symmetic weights between hidden variables

        if predictor is not None:
            self.predictor = predictor

        self.env = gym.make('FrozenLake8x8-v0')
        self.observation = self.env.reset()
        self.maxobs = 64   # !!!!!! For frozen Lake
        self.numact = 4
        
        self.defaultGamma = 0.999  # as in Otsuka - Solving POMPDs
        self.defaultLr = 0.01  # as in Otsuka - Solving POMPDs

    # Transorm the sensor/motor input into integer index
    def InputToIndex(self, x, xmax, bitsize):
        return int(np.floor((x + xmax) / (2 * xmax + 10 * np.finfo(float).eps) * 2**bitsize))
       
    # Create state nodes for ESN to be observation from env + predictor's memory
    # observation is binarised {0; 1}, linear memory nodes are used directly
    def createJointInput(self, state, memory):
        inp = np.zeros(self.Inpsize)
        inp[:self.Ssize] = bitfield(self.InputToIndex(state, self.maxobs, self.Ssize), self.Ssize) # binarise observation
        inp[self.Ssize:self.Inpsize] = memory  # use memory directly - better performance
        return inp
           
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
        
        return float(a+b+c+d+e+f)
    
    def SarsaLearning(self, total_episodes, max_steps, Beta, gamma=None, lr=None):
        
#         pdb.set_trace()

        self.predictor.setHistory(total_episodes, max_steps)  # create history array 
        self.log = np.tile(np.repeat(-1.0,7),(total_episodes, 1))  # track learning
        
        for episode in range(total_episodes):
            
            beta = Beta[episode]
            
            #memory is not reset per episode
            if episode == 0:
                memory = self.predictor.laststate
            else:
                memory = memory2
            
            # reset the environment
            state = self.env.reset()

            # choose a (action) based on s = obs+m (obs+memory)
            state_memory = self.createJointInput(state, memory)
            action = self.ChooseAction(state_memory, beta)
           
            # calculate Q(s,a)
            Q1 = -1*self.CalcFreeEnergy(state_memory, action)

            self.initialiseDeltas()  # update of senhid weights, hidmot weights and the biases for the sen, mot and hid
            Qs = np.repeat(-np.inf, max_steps)
            PredQual = np.repeat(-np.inf, max_steps)
            int_rew = np.repeat(-np.inf, max_steps)
            
            t = 0
            while t < max_steps:             
                
                # calculate m'
#                esn_input = np.hstack([state_bit, action_bit]).reshape(1,-1)
                esn_input = np.array([state, action]).reshape(1,-1)
                memory2 = self.predictor.get_states(esn_input, extended=False, continuation=True) # we only take state activations, not concatenate states with input which is done to train the weights and to also predict                                                                           
                
                # step with a and receive obs'
                state2, ext_reward, done, info = self.env.step(action)
                
                #--
                
                # Predictor improvement and calculation of internal reward
                self.predictor.history[episode,t,:] = np.array([state, action, state2, ext_reward])  # update predictor history
                int_reward, qualp = self.predictor.calculateInternalReward() # calculate internal reward
            
                #### testing by setting internal reward to zero
                int_reward = 0
                # Calculation of reward for SARSA update
                reward = int_reward+ext_reward
                
                #--
                
                # choose a' based on s' = obs'+m'
                state_memory2 = self.createJointInput(state2, memory2) # add the predictor's state to the obsrvations
                action2 = self.ChooseAction(state_memory2, beta)
                
                # calculate Q(s',a')
                Q2 = -1*self.CalcFreeEnergy(state_memory2, action2) # calculate Q2 = Q(s',a') = -F(s',a')
                
                # calculate update to the weights of the network currently having s and a
                self.SarsaUpdateOnline(Q1, reward, Q2, state_memory, action, gamma, lr)
                
                Qs[t] = Q1
                PredQual[t] = qualp
                int_rew[t] = int_reward

                # updates for loop
                state_memory = state_memory2
                state = state2  # update obs = obs'   (needed to calculate m' = obs + a)
                action = action2  # update a = a'  (needed to calculate m' = obs + a and to also get obs')
                Q1 = Q2  # no need to update m (memory), because Q1 is directly updated

                t += 1
                
                if done:
                    break
                
            # apply episodic update rule: t+1 is the number of time steps out of the max actually performed
            self.SarsaUpdateEpisodic(t)
            
            # update log
            vishidJ = np.vstack((self.J[:self.Inpsize,self.Inpsize:-self.Msize], np.transpose(self.J[self.Inpsize:-self.Msize, -self.Msize:])))
            self.log[episode, :] = np.array([state, ext_reward, np.mean(PredQual[:t]), np.mean(Qs[:t]), np.mean(int_rew[:t]),
                    np.mean(np.abs(vishidJ)), np.max(np.abs(vishidJ))])
            

    def SarsaUpdateOnline(self, Q1, reward, Q2, state_bit, action, gamma, lr):
        
#         pdb.set_trace()
        
        if gamma is None:
            gamma = self.defaultGamma
        if lr is None:
            lr = self.defaultLr
        
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
    def ChooseAction(self, state, beta):
        
#         pdb.set_trace()

        # calculate probabilities of all actions - based on Otsuka's paper
        p_a = np.zeros((self.env.nA,2))
        for a in range(self.env.nA):
            bfe = 700 if -beta*self.CalcFreeEnergy(state, a) > 700 else -beta*self.CalcFreeEnergy(state, a)
            fa = np.exp(bfe)
            fa_ = 0
            for a_ in range(self.env.nA):
                bfe = 700 if -beta*self.CalcFreeEnergy(state, a_) > 700 else -beta*self.CalcFreeEnergy(state, a_)
                fa_ = fa_ + np.exp(bfe)
            p_a[a,1] = fa/fa_
            p_a[a,0] = a  #add index column

        # sample an action from the distribution
        ord_p_a = p_a[p_a[:,1].argsort()]
        ord_p_a[:,1] = np.cumsum(ord_p_a[:,1])

        b = np.array(ord_p_a[:,1] > np.random.rand())
        act = int(ord_p_a[b.argmax(),0])  # take the index of the chosen action
 
        return act
    
    def displayRunData(self, total_episodes, num_el):

        res = self.log
    
        # pred errors
    
        x = range(total_episodes)
        y = res[:,2]
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label='mean predictor error')
    
        ax.legend()
        plt.show()
    
        # int reward
    
        y = res[:,4]
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label='mean internal reward')
    
        ax.legend()
        plt.show()
    
        # actor -free energy
    
        y = res[:,3]
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label='mean negative free energy')
    
        ax.legend()
        plt.show()
    
        # mean J
    
        y = res[:,5]
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label='mean J')
    
        ax.legend()
        plt.show()
    
        # max J
    
        y = res[:,6]
    
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y, label='max J')
    
        ax.legend()
        plt.show()

        
        
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

