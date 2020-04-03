"""
Code for ising taken from Aguilera's code for "Adaptation to Criticality":
Modifications:
- Restricted - connections within hidden layer not being updated
- Embodiment for environment Frozen Lake
- RL Learning by a variant of SARSA - to Hinton's paper "Using Q-energies"

** Sensors and motors 0/1 instead of -1/1

"""


import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import gym
import math

import pdb


class ising:
    # Initialize the network
    def __init__(self, netsize, Nsensors=1, Nmotors=1):  # Create ising model

#         pdb.set_trace()
        
        self.size = netsize		#Network size
        self.Ssize = Nsensors  # Number of sensors
        self.Msize = Nmotors  # Number of motors

        self.h = np.zeros(netsize) # local biases
        self.J = np.zeros((netsize, netsize)) # symmetic weights between hidden variables
        self.max_weights = 2          # why do we need to restrict the weights to a maximum value?

        self.randomize_state()

        self.env = gym.make('FrozenLake8x8-v0')
        self.observation = self.env.reset()

        self.BetaCrit = 1.0  
        self.defaultT = max(100, netsize * 20)
        
        self.defaultGamma = 0.999  # as in Otsuka - Solving POMPDs
        self.defaultLr = 0.01  # as in Otsuka - Solving POMPDs

        self.Ssize1 = 0
        
        self.rewardsPerEpisode = 0     #keep track of rewards
        self.successfulEpisodes = 0
        self.observations = np.repeat(-1,1000*5000)    #keep track of reached states
        
      
    def initialise_wiring(self):
        self.J = np.random.normal(0, 0.1, (self.size, self.size))

    def get_state(self, mode='all'):
        if mode == 'all':
            return self.s
        elif mode == 'motors':
            return self.motors
        elif mode == 'sensors':          # isn't mode sensors the same as input??
            return self.s[0:self.Ssize]
        elif mode == 'input':
            return self.sensors
        elif mode == 'non-sensors':
            return self.s[self.Ssize:]
        elif mode == 'hidden':
            return self.s[self.Ssize:-self.Msize]

    # gets the index of the configuration of the neurons - state as one number
    def get_state_index(self, mode='all'):
        return bool2int(0.5 * (self.get_state(mode) + 1))

    # Randomize the state of the network
    def randomize_state(self):
        self.s = np.random.randint(0, 2, self.size) * 2 - 1           # make units -1 or 1 
        self.sensors = np.random.randint(0, 2, self.Ssize) * 2 - 1    # make sensors -1 or 1
        self.motors = np.random.randint(0, 2, self.Msize) * 2 - 1 

    # Randomize the position of the agent
    def randomize_position(self):
        self.observation = self.env.reset()

    # Set random bias to sets of units of the system
    # bias to hidden and action
    def random_fields(self, max_weights=None):
        if max_weights is None:
            max_weights = self.max_weights
        self.h[self.Ssize:] = max_weights * \
            (np.random.rand(self.size - self.Ssize) * 2 - 1)

    # Set random connections to sets of units of the system
    def random_wiring(self, max_weights=None):  # Set random values for J
        if max_weights is None:
            max_weights = self.max_weights
        for i in range(self.size):
            for j in np.arange(i + 1, self.size):
                if i < j and (i >= self.Ssize or j >= self.Ssize):  # don't add connections between the sensors
                    # what about the motors? connection between the two will be added: doesn't matter because the correlations are not considered?
                    self.J[i, j] = (np.random.rand(1) * 2 - 1) * self.max_weights
        # get rid of connections between hidden units, and also motor (still dont know why this hasnt been done)
        self.J[self.Ssize:-self.Msize,self.Ssize:-self.Msize] = 0  # between hidden
        self.J[:self.Ssize,-self.Msize:] = 0             # between sensor and motor
        self.J[-self.Msize:,-self.Msize:] = 0            # between motor

    # Update the position of the agent
    def Move(self):
        action = int(np.digitize(
            np.sum(self.s[-self.Msize:]) / self.Msize, [-2, -1/2, 0, 1/2, 2])) - 1
        observation, reward, done, info = self.env.step(action)
        
        self.observation = observation  # update latest observation
        
        self.rewardsPerEpisode += reward    #update rewards per episode
        self.observations[(self.observations == -1).argmax()] = observation      #add to list woth visited states

    # Update the state of the sensor
    def UpdateSensors(self, state=None):
        if state is None:
            self.sensors = 2 * bitfield(self.observation, self.Ssize) - 1
        else:
            self.sensors = bitfield(state, self.Ssize) 
        
    # Update the state of the motor?
    def UpdateMotors(self, action):
        self.motors = bitfield(action, self.Msize)

    # Execute step of the Glauber algorithm to update the state of one unit
    def GlauberStep(self, i=None): 
        if i is None:
            i = np.random.randint(self.size)

        I = 0
        if i < self.Ssize:
            I = self.sensors[i]
        eDiff = 2 * self.s[i] * (self.h[i] + I +
                                 np.dot(self.J[i, :] + self.J[:, i], self.s))
        if eDiff * self.BetaCrit < np.log(1 / np.random.rand() - 1):    # Glauber
            self.s[i] = -self.s[i]

    # Update random unit of the agent
    def Update(self, i=None):
        if i is None:
            i = np.random.randint(-1, self.size)
        if i == -1:
            self.Move()
            self.UpdateSensors()
        else:
            self.GlauberStep(i)

    # Sequentially update state of all units of the agent in random order
    def SequentialUpdate(self):
        for i in np.random.permutation(range(-1, self.size)):
            self.Update(i)

    # Step of the learning algorith to ajust correlations to the critical regime
    def AdjustCorrelations(self, T=None):
        if T is None:
            T = self.defaultT

        self.m = np.zeros(self.size)
        self.c = np.zeros((self.size, self.size))
        self.C = np.zeros((self.size, self.size))

        # Main simulation loop:
        self.x = np.zeros(T)      # to store the positions of the car during all T
        samples = []
        for t in range(T):

            self.SequentialUpdate()
            self.x[t] = self.observation
            self.m += self.s
            for i in range(self.size):
                self.c[i, i + 1:] += self.s[i] * self.s[i + 1:]
        self.m /= T
        self.c /= T
        for i in range(self.size):
            self.C[i, i + 1:] = self.c[i, i + 1:] - self.m[i] * self.m[i + 1:]

        c1 = np.zeros((self.size, self.size))
        for i in range(self.size):
            inds = np.array([], int)
            c = np.array([])
            for j in range(self.size):
                if not i == j:
                    inds = np.append(inds, [j])
                if i < j:
                    c = np.append(c, [self.c[i, j]])
                elif i > j:
                    c = np.append(c, [self.c[j, i]])
            order = np.argsort(c)[::-1]
            c1[i, inds[order]] = self.Cint[i, :]
        self.c1 = np.triu(c1 + c1.T, 1)
        self.c1 *= 0.5

        self.m[0:self.Ssize] = 0          
        self.m1[0:self.Ssize] = 0     #sensors have objective mean 0 but in the paper they say it's all of the units but the sensors that have mean 0??
        self.c[0:self.Ssize, 0:self.Ssize] = 0    #set corr in between sensors to 0
        self.c[-self.Msize:, -self.Msize:] = 0    #set corr in between motors to 0
        self.c[0:self.Ssize, -self.Msize:] = 0    #set corr between sensors and motors to 0
        self.c1[0:self.Ssize, 0:self.Ssize] = 0
        self.c1[-self.Msize:, -self.Msize:] = 0
        self.c1[0:self.Ssize, -self.Msize:] = 0
        
        # make it restricted BM
        self.c[self.Ssize:-self.Msize,self.Ssize:-self.Msize] = 0   #set corr in between hidden units to 0
        self.c1[self.Ssize:-self.Msize,self.Ssize:-self.Msize] = 0   #for reference as well
        
        dh = self.m1 - self.m
        dJ = self.c1 - self.c

        return dh, dJ

    # Algorithm for poising an agent in a critical regime
    def CriticalLearning(self, Iterations, T=None):
        u = 0.01
        count = 0
        dh, dJ = self.AdjustCorrelations(T)
        for it in range(Iterations):
            count += 1
            self.h += u * dh
            self.J += u * dJ

            if it % 10 == 0:
                self.randomize_state()
                self.randomize_position()
                
                if self.rewardsPerEpisode >= 1:     # keep track of the times the agent reached the goal
                    self.successfulEpisodes += 1
                self.rewardsPerEpisode = 0
                
            Vmax = self.max_weights
            for i in range(self.size):
                if np.abs(self.h[i]) > Vmax:        # why do we need to restrict the weights and biases to a maximum value?
                    self.h[i] = Vmax * np.sign(self.h[i])
                for j in np.arange(i + 1, self.size):
                    if np.abs(self.J[i, j]) > Vmax:
                        self.J[i, j] = Vmax * np.sign(self.J[i, j])

            dh, dJ = self.AdjustCorrelations(T)
           
    # Calculate hat_h_k for each k (expert)    
    def ExpectedValueExperts(self, sensors, motors):
        # self.s[Ssize:-Msize] - all hidden
        # self.h[Ssize:-Msize] - biases of hidden
        # self.J[:Ssize,Ssize:-Msize] - sensor-hidden connections
        # self.J[Ssize:-Msize, -Msize:] - hidden-motor conn.
        
        num_e = self.size - (self.Ssize+self.Msize)  # number of hidden units (experts)
        ve = np.zeros(num_e)             # array of expected values for experts
        for k in range(num_e):
            ve[k] = sigmoid(np.dot(self.J[0:self.Ssize,self.Ssize:-self.Msize][:,k], sensors) + 
                            np.dot(self.J[self.Ssize:-self.Msize, -self.Msize:][k,:], motors) +
                            self.h[self.Ssize:-self.Msize][k])
        
        return ve
    
    # takes in state and action and uses their binary equivalents to calculate the energy (not the actual network's)
    def CalcFreeEnergy(self, state, action):
        
#         pdb.set_trace()
        
        sensors = bitfield(state, self.Ssize) #simulate sensors,motors of the network for the state, action
        motors = bitfield(action, self.Msize)
        
        ve = self.ExpectedValueExperts(sensors, motors).reshape(-1,1)  #calculate expected hidden values
        ss = sensors.reshape(1,-1)
        sm = motors.reshape(1,-1) 
        
        # calculate the negative log-likelihood
        a = -np.dot(np.dot(ss, self.J[:self.Ssize,self.Ssize:-self.Msize]), ve)    # 1
        b = -np.dot(ss, self.h[:self.Ssize].reshape(-1,1))                             # 2
        c = -np.dot(np.dot(sm, np.transpose(self.J[self.Ssize:-self.Msize, -self.Msize:])), ve)  # 3
        d = -np.dot(sm, self.h[-self.Msize:].reshape(-1,1))                            # 4
        e = -np.dot(self.h[self.Ssize:-self.Msize], ve)                       # 5
        
        # calculate the negative entropy
        f = 0                                                           # 6 + 7
        for k in range(len(ve)):
            f += ve[k]*np.log(ve[k]) + (1 - ve[k])*np.log(1 - ve[k])
        
        return (a+b+c+d+e+f)
    
    def SarsaLearning(self, total_episodes, max_steps, Beta, gamma=None, lr=None):
        
#        pdb.set_trace()
        
        self.rewards = np.zeros(total_episodes)
        
        for episode in range(total_episodes):
            
            beta = Beta[episode]
            
            state = self.env.reset()
            self.UpdateSensors(state)

            action = self.ChooseAction(state, beta)
            self.UpdateMotors(action)
           
            Q1 = -1*self.CalcFreeEnergy(state, action)   #calculate Q1 = Q(s,a) = -F(s,a)

            t = 0
            while t < max_steps:
                
#                self.env.render()
                
                state2, reward, done, info = self.env.step(action)

                action2 = self.ChooseAction(state2, beta)
                
                Q2 = -1*self.CalcFreeEnergy(state2, action2)        #calculate Q2 = Q(s',a') = -F(s',a')

                self.SarsaUpdate(Q1, reward, Q2, gamma, lr)

                #update the network
                self.UpdateSensors(state2)
                self.UpdateMotors(action2)
                action = action2   # no need to update state, because not needed - Q1 is directly calculated
                Q1 = Q2

                t += 1

                if done:
                    break
                    
            self.rewards[episode] = reward
    
    # works with the network's actual sensors and motors
    def SarsaUpdate(self, Q1, reward, Q2, gamma, lr):
        
        if gamma is None:
            gamma = self.defaultGamma
        if lr is None:
            lr = self.defaultLr
        
        rDiff = lr*(reward + gamma * Q2 - Q1)
        
        # calculate dQ/Wsensors
        ve = self.ExpectedValueExperts(self.sensors, self.motors).reshape(1,-1)
        ve_new = np.repeat(ve, len(self.sensors)).reshape(-1,1)
        s_ne = np.repeat(self.sensors.reshape(1,-1), ve.shape[1], axis=0)
        s_new = s_ne.reshape(s_ne.shape[0]*s_ne.shape[1],1)
        dQdWs = np.multiply(ve_new,s_new)
        
        # calculate dQ/Wmotors
        ve_new = np.repeat(ve, len(self.motors)).reshape(-1,1)
        m_ne = np.repeat(self.motors.reshape(1,-1), ve.shape[1], axis=0)
        m_new = m_ne.reshape(m_ne.shape[0]*m_ne.shape[1],1)
        dQdWm = np.multiply(ve_new,m_new)
        
        # update weights of sensors and motors
        num_h = self.size - (self.Ssize+self.Msize)
        
        dWs = (rDiff*dQdWs).reshape(num_h, self.Ssize)
        dWm = (rDiff*dQdWm).reshape(num_h, self.Msize)
        
        self.J[:self.Ssize,self.Ssize:-self.Msize] = np.transpose(np.transpose(self.J[:self.Ssize,self.Ssize:-self.Msize]) + dWs)
        self.J[self.Ssize:-self.Msize, -self.Msize:] = self.J[self.Ssize:-self.Msize, -self.Msize:] + dWm
        
        # update biases
        #!! not in the paper - I think this should be it??
        self.h[:self.Ssize] = self.h[:self.Ssize] + rDiff*self.sensors
        self.h[-self.Msize:] = self.h[-self.Msize:] + rDiff*self.motors
        self.h[self.Ssize:-self.Msize] =  self.h[self.Ssize:-self.Msize] + rDiff*ve
        
        
    def ChooseAction(self, state, beta):
        
        try:
            # calculate probabilities of all actions - based on Otsuka's paper
            p_a = np.zeros((self.env.nA,2))
            for a in range(self.env.nA):
                fa = np.exp(-beta*self.CalcFreeEnergy(state, a))
                fa_ = 0
                for a_ in range(self.env.nA):
                    fa_ = fa_ + np.exp(-beta*self.CalcFreeEnergy(state, a_))
                p_a[a,1] = fa/fa_
                p_a[a,0] = a  #add index column

            # sample an action from the distribution
            ord_p_a = p_a[p_a[:,1].argsort()]
            ord_p_a[:,1] = np.cumsum(ord_p_a[:,1])

            b = np.array(ord_p_a[:,1] > np.random.rand())
            act = int(ord_p_a[b.argmax(),0])  # take the index of the chosen action
        
        except RuntimeWarning:
            pdb.set_trace()
 
        return act
        
        
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

