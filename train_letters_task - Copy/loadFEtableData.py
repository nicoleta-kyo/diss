# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:01:38 2020

@author: Niki
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import string

def calc_mean_qual(eps, qual, step):
    vals = len([i for i in range(0,eps,step)])  
    quals = np.zeros(vals)
    for i in range(vals):
        if i==(vals-1):
            quals[i] = np.mean(qual[step*i:eps])
        quals[i] = np.mean(qual[step*i:step*(i+1)])
    return quals

def calc_cum_q(eps, qual,step):
    vals = len([i for i in range(0,eps,step)])  
    quals = np.zeros(vals)
    for i in range(vals):
        if i==(vals-1):
            quals[i] = np.mean(qual[step*i:eps])
        quals[i] = np.mean(qual[step*i:step*(i+1)])
    return quals

file = 'runData\\rbm\\noncrit_rbm_noncrit_esn.pkl'

with open(file, 'rb') as inp:
    resdata = pickle.load(inp) 
    
fet = resdata['i_fe']

#plot the change of Q for each state
eps=5000

#stNfe = fet[:eps,13,3] 
#mstNfe = calc_mean_qual(eps,stNfe)

x=range(len(mstAfe))
y1=mstAfe
l1='state A Q-value'
y2=mstNfe
l2='Goal state Q-value'


fig, ax = plt.subplots()
line1, = ax.plot(x, -y1, ,label=l1)
line1, = ax.plot(x, -y1, ,label=l1)
line1, = ax.plot(x, -y1, ,label=l1)
line1, = ax.plot(x, -y1, ,label=l1)


line2, = ax.plot(x, -y2, ,label=l2)
ax.legend()
#ax.set_ylim([0,0.01])
ax.set_xlabel('episode')
ax.set_ylabel('free energy')

# all states

#mean

st=5453
end=5600
eps=end-st
step=1
stfes = np.zeros((14,int(eps/step)))
for i in range(14):
    stfes[i] = -1*calc_mean_qual(eps,fet[st:end,i,3],step)

## cumsum
#stfes = np.zeros((14,eps))
#for i in range(14):
#    stfes[i] = -1*np.cumsum(fet[:eps,i,3])   

fig, ax = plt.subplots()
#for i in range(14):
#    lab=str(i)
#    line, = ax.plot(x, stfes[i],label=lab)

x=range(stfes.shape[1])
#x=range(50)
line1, = ax.plot(x, stfes[0], 'black', label='A')
#line1, = ax.plot(x, fet[:50,0,3], label='0')
#line2, = ax.plot(x, stfes[1], label='1')
#line3, = ax.plot(x, stfes[2], label='2')
#line4, = ax.plot(x, stfes[3], label='3')
#line5, = ax.plot(x, stfes[4], label='4')
#line6, = ax.plot(x, stfes[5], label='5')
#line7, = ax.plot(x, stfes[6], label='6')
#line8, = ax.plot(x, stfes[7], 'dimgrey', label='H')
#line9, = ax.plot(x, stfes[8], 'dimgrey', label='I')
#line10, = ax.plot(x, stfes[9],  'gray', label='J')
#line11, = ax.plot(x, stfes[10],  'darkgrey', label='K')
#line12, = ax.plot(x, stfes[11],  'silver', label='L')
#line13, = ax.plot(x, stfes[12], 'lightgrey', label='M')
#line14, = ax.plot(x, stfes[13], 'whitesmoke', label='N')
#line14, = ax.plot(x,  fet[:50,13,3],  label='13')

ax.set_xticklabels(['0', '0','1000', '2000', '3000', '4000', '5000'])
ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('Q-value')


# cum Q

fig, ax = plt.subplots()

cumAfe = fet[:eps,13,3] 
c=np.cumsum(mstAfe)
x=range(len(c))
line1, = ax.plot(x, c, label='0')


#
stend = fet[5450:5599,0,3]
x=range(len(stend))
fig, ax = plt.subplots()
line1, = ax.plot(x, stend, label='0')

# last few eps

fig, ax = plt.subplots()
st0end = fet[5450:5599,0,3] 
line1, = ax.plot(range(len(st0end)), st0end, label='0')

fig, ax = plt.subplots()
st0end = fet[5450:5599,0,1] 
line1, = ax.plot(range(len(st0end)), st0end, label='0')

fig, ax = plt.subplots()
st0end = fet[5450:5599,0,2] 
line1, = ax.plot(range(len(st0end)), st0end, label='0')

#
stend = fet[5450:5599,0,3]
x=range(len(stend))
fig, ax = plt.subplots()
line1, = ax.plot(x, stend, label='0')

## heat map of the visited states

arr=np.zeros((5600,14),dtype='int')
for i in range(5600):
    arr[i,:] = fet[i,:,0]
arr=arr[:100,:]
arr=arr.T

eps=5600
step=100
arr = np.zeros((14,int(eps/step)),dtype='int')
for i in range(14):
    arr[i] = calc_mean_qual(eps,fet[:eps,i,0],step)

fig, ax = plt.subplots()
im = ax.imshow(arr, aspect=3)

stlabs=[string.ascii_uppercase[i] for i in range(14)]
ax.set_yticks(np.arange(len(stlabs)))
#ax.set_yticks(np.arange(len(combslabs)))
ax.set_yticklabels(stlabs)
ax.set_xticklabels([str(i*10) for i in range(-100,eps,step)])
#
## Create colorbar
#cbar = ax.figure.colorbar(im, ax=ax)
#cbar.ax.set_ylabel('Number of time-steps', rotation=-90, va="bottom")

##maptitle = 'Coverage of states'
#ax.set_title(maptitle)


eps=1100
step=100
arr = np.zeros((14,int(eps/step)),dtype='int')
for i in range(14):
    arr[i] = calc_mean_qual(eps,fet[4500:5600,i,0],step)

fig, ax = plt.subplots()
im = ax.imshow(arr, aspect=1)

stlabs=[string.ascii_uppercase[i] for i in range(14)]
ax.set_yticks(np.arange(len(stlabs)))
#ax.set_yticks(np.arange(len(combslabs)))
ax.set_yticklabels(stlabs)
ax.set_xticklabels([str(i) for i in range(4300,5600,step*2)])

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Number of time-steps', rotation=-90, va="bottom")

maptitle = 'Coverage of states'
ax.set_title(maptitle)

## goal only - fe
start=0
end=500

y1=fet[start:end,0,3]
y2=fet[start:end,1,3]
y3=fet[start:end,3,3]
y4=fet[start:end,4,3]
y5=fet[start:end,51,3]

x=range(end-start)

fig, ax = plt.subplots()
line1, = ax.plot(x, -1*y1, 'black', label='A')
line2, = ax.plot(x, -1*y2, label='B')
line3, = ax.plot(x, -1*y3, label='C')
line4, = ax.plot(x, -1*y4, label='D')
line5, = ax.plot(x, -1*y5, label='Goal')


ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('Q-value')

## goal only - entropy

start=0
end=500

y1=fet[start:end,0,2]

x=range(end-start)

fig, ax = plt.subplots()
line1, = ax.plot(x, -1*y1, 'black', label='Goal state')


ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('entropy')

## goal only - entropy

start=0
end=500

y1=fet[start:end,0,1]

x=range(end-start)

fig, ax = plt.subplots()
line1, = ax.plot(x, -1*y1, 'black', label='Goal state')


ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('expected energy')


