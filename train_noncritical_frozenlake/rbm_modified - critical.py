"""
Code for ising taken from Aguilera's code for "Adaptation to Criticality":
Changes:
- Restricted - connections within hidden layer not being updated
- Embodiment for environment Frozen Lake
- RL Learning by a variant of SARSA - to Hinton's paper "Using Q-energies"
     *using sensors and motors variables (not the state variable)
     
For Otsuka's model architecture:
- Predictor ESN as parameter so it can have access to its reservoir and history
- Memory layer: doesn't get reset throughout training
- Inputs are binarised observation, memory, reward
- An episode is T*10 time steps

Intrinsic Motivation:
- internal reward is taken only if there has been improvement in the predictor, and it is bounded from 0 to 1

""" 


import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import gym
import time

import pdb


class ising:
    # Initialize the network
    def __init__(self, netsize, Nmemory, Nreward, Nsensors=1, Nmotors=1, predictor=None):  # Create ising model

#         pdb.set_trace()
        
        self.size = netsize		#Network size
        self.Ssize = Nsensors  # Number of sensors
        self.Msize = Nmotors  # Number of motors
        self.Memsize = Nmemory # Number of memory units
        self.Rewsize = Nreward # Number of units encoding reward
        self.Inpsize = Nsensors + Nmemory + Nreward  # Number of sensors+memory+reward

        self.h = np.zeros(netsize) # local biases
        self.J = np.zeros((netsize, netsize)) # symmetic weights between hidden variables
        self.max_weights = 2          # prevent (one of) the weights from growing too much

        if predictor is not None:
            self.predictor = predictor

        self.env = gym.make('FrozenLake8x8-v0')
        self.observation = self.env.reset()
        self.maxobs = 64   # !!!!!! For frozen Lake
        self.maxact = 4
        self.maxrew = 3
        
        self.randomize_state() # randomise states s
        self.initialise_sensors() # initialise sensors to point to start observation and to empty memory of the predictor

        self.BetaCrit = 1.0  
        self.defaultT = max(100, netsize * 20)
        
        
    def get_state(self, mode='all'):
        if mode == 'all':
            return self.s
        elif mode == 'motors':
            return self.motors
        elif mode == 'sensors': 
            return self.s[0:self.Inpsize]
        elif mode == 'input':
            return self.sensors
        elif mode == 'non-sensors':
            return self.s[self.Inpsize:]
        elif mode == 'hidden':
            return self.s[self.Inpsize:-self.Msize]

    # gets the index of the configuration of the neurons - state as one number
    def get_state_index(self, mode='all'):
        return bool2int(0.5 * (self.get_state(mode) + 1))

    def initialise_sensors(self):
        self.sensors = np.repeat(-1, self.Ssize+self.Memsize+self.Rewsize)
        self.sensors[:self.Ssize] = bitfield(self.InputToIndex(self.observation, self.maxobs, self.Ssize), self.Ssize) * 2 - 1
    
    # Randomize the state of the network
    def randomize_state(self):     
        self.s = np.random.randint(0, 2, self.size) * 2 - 1           # make units -1 or 1     
    
    # Randomize the position of the agent
    def randomize_position(self):
        self.observation = self.env.reset()

    # Set random bias to sets of units of the system
    # bias to hidden and action
    def random_fields(self, max_weights=None):
        if max_weights is None:
            max_weights = self.max_weights
        self.h[self.Inpsize:] = max_weights * \
            (np.random.rand(self.size - self.Inpsize) * 2 - 1)

    # Set random connections to sets of units of the system
    def random_wiring(self, max_weights=None):  # Set random values for J
        if max_weights is None:
            max_weights = self.max_weights
        for i in range(self.size):
            for j in np.arange(i + 1, self.size):
                if i < j and (i >= self.Inpsize or j >= self.Inpsize):  # don't add connections between the sensors
                    # what about the motors? connection between the two will be added: doesn't matter because the correlations are not considered?
                    self.J[i, j] = (np.random.rand(1) * 2 - 1) * self.max_weights
        # get rid of connections between hidden units, and also motor (still dont know why this hasnt been done)
        self.J[self.Inpsize:-self.Msize,self.Inpsize:-self.Msize] = 0  # between hidden
        self.J[:self.Inpsize,-self.Msize:] = 0             # between sensor and motor
        self.J[-self.Msize:,-self.Msize:] = 0            # between motor

    # Update the position of the agent
    def Move(self):
        
#         pdb.set_trace()
        
        # step in environment and collect state2 and ext_reward
        action = int(np.digitize(np.sum(self.s[-self.Msize:]) / self.Msize, [-1.1, -1/2, 0, 1/2, 1.1])) - 1
        state2, ext_reward, done, info = self.env.step(action)
        
        # update predictor's history and collect its memory
        self.predictor.history[self.episode,self.t,:] = np.array([self.observation, action, state2, ext_reward])  # update predictor history
        # m' = f(s, a)  
        memory = self.predictor.get_states(np.array([self.observation, action]).reshape(1,-1), extended=False, continuation=True)
        
        # calculate internal reward; pass int+ext to the rbm
        int_reward, qual = self.predictor.calculateInternalReward()
        self.log[self.episode, self.t, 0] = qual # store predictor's error after seeing the output of this move 
        
        reward = int_reward+ext_reward
        
        # new state = state2 + memory (from state,action) + reward (int + ext)
        self.observation = state2  
        self.UpdateSensors(state2,memory,reward)
    
    # Transorm the sensor/motor input into integer index
    def InputToIndex(self, x, xmax, bitsize):
        return int(np.floor((x + xmax) / (2 * xmax + 10 * np.finfo(float).eps) * 2**bitsize))
    
    # Update the state of the sensor
    def UpdateSensors(self, state, memory, reward):  
        
        # binarise observation
        state_bit = bitfield(self.InputToIndex(state, self.maxobs, self.Ssize), self.Ssize) * 2 - 1 
        self.sensors[:self.Ssize] = state_bit
        # binarise memory
        memory_bit = np.zeros((1, memory.shape[1]))
        for i in range(memory.shape[1]):
            memory_bit[0,i] = -1 if memory[0,i] < 0.5 else 1
        self.sensors[self.Ssize:(self.Ssize+self.Memsize)]= memory_bit
        # binarise reward
        reward_bit = bitfield(self.InputToIndex(reward, self.maxrew, self.Rewsize), self.Rewsize) * 2 - 1
        self.sensors[(self.Ssize+self.Memsize):self.Inpsize]= reward_bit
        
    # Updates only the observation of the sensor, the memory and reward are kept the same
    def resetSensors(self, state):  
        
        state_bit = bitfield(self.InputToIndex(state, self.maxobs, self.Ssize), self.Ssize) * 2 - 1 
        self.sensors[:self.Ssize] = state_bit

    # Execute step of the Glauber algorithm to update the state of one unit
    def GlauberStep(self, i=None): 
        if i is None:
            i = np.random.randint(self.size)

        I = 0
        if i < self.Inpsize:
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
        for t in range(T):
            
            self.SequentialUpdate()
            self.x[t] = self.observation
            self.m += self.s
            for i in range(self.size):
                self.c[i, i + 1:] += self.s[i] * self.s[i + 1:]
            
            self.t += 1
            
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

        self.m[0:self.Inpsize] = 0          
        self.m1[0:self.Inpsize] = 0     #sensors have objective mean 0 but in the paper they say it's all of the units but the sensors that have mean 0??
        self.c[0:self.Inpsize, 0:self.Inpsize] = 0    #set corr in between sensors to 0
        self.c[-self.Msize:, -self.Msize:] = 0    #set corr in between motors to 0
        self.c[0:self.Inpsize, -self.Msize:] = 0    #set corr between sensors and motors to 0
        self.c1[0:self.Inpsize, 0:self.Inpsize] = 0
        self.c1[-self.Msize:, -self.Msize:] = 0
        self.c1[0:self.Inpsize, -self.Msize:] = 0
        
        # make it restricted BM
        self.c[self.Inpsize:-self.Msize,self.Inpsize:-self.Msize] = 0   #set corr in between hidden units to 0
        self.c1[self.Inpsize:-self.Msize,self.Inpsize:-self.Msize] = 0   #for reference as well
        
        dh = self.m1 - self.m
        dJ = self.c1 - self.c

        return dh, dJ

    # Algorithm for poising an agent in a critical regime
    def CriticalLearning(self, Iterations, T=None):
        
#         pdb.set_trace()
        tic = time.perf_counter()
    
        self.observations = np.repeat(-1,Iterations*T)    #keep track of reached states
        self.predictor.setHistory(int(Iterations/10)+1, T*10 ) # resetting the env every 10 iterations, so episodes and steps adjusted accordingly 
        self.log = np.repeat(-np.inf,(int(Iterations/10)+1)*T*10).reshape(int(Iterations/10)+1, T*10, 1) # store pred quality for every time step
        
        self.t = 0
        self.episode = 0
        
        u = 0.01
        count = 0
        dh, dJ = self.AdjustCorrelations(T)
        for it in range(Iterations):
            
            count += 1
            self.h += u * dh
            self.J += u * dJ

            if it % 10 == 0:
                
                toc = time.perf_counter()
                print('Episode ' + str(self.episode) + ' took ' + str(int((toc - tic)/60)) + ' minutes.')
                
                tic = time.perf_counter()
#                 pdb.set_trace()
                
                self.randomize_position()  # updates self.observation to be 0
                
                # update the sensors with observation=0, the memory and the reward are kept the same
                self.resetSensors(self.observation)
                self.randomize_state()
                
                print()
                
                self.episode += 1  # first episode will have T instead of T*10 entries but thats ok because the predictor's reward method only takes actual entries
                self.t = 0

                
            Vmax = self.max_weights
            for i in range(self.size):
                if np.abs(self.h[i]) > Vmax:        # why do we need to restrict the weights and biases to a maximum value?
                    self.h[i] = Vmax * np.sign(self.h[i])
                for j in np.arange(i + 1, self.size):
                    if np.abs(self.J[i, j]) > Vmax:
                        self.J[i, j] = Vmax * np.sign(self.J[i, j])

            dh, dJ = self.AdjustCorrelations(T)

        
        
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
