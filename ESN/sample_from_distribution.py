# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:13:02 2020

@author: Niki
"""

import numpy as np

arr = [0.2, 0.3, 0.5]
prob_n = np.asarray(arr)
csprob_n = np.cumsum(prob_n)

for i in range(20):
    
    
    a = np.random.rand()
    print("a: "+ str(a))
    b = np.array(csprob_n > a)
    print("b: " + str(b))
    c = b.argmax()
    print("c: " + str(c))





import matplotlib.pyplot as plt
import numpy as np

# create some randomly ddistributed data:
data = np.random.randn(10000)

# sort the data:
data_sorted = np.sort(data)

# calculate the proportional values of samples
p = 1. * np.arange(len(data)) / (len(data) - 1)

# plot the sorted data:
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(p, data_sorted)
ax1.set_xlabel('$p$')
ax1.set_ylabel('$x$')

ax2 = fig.add_subplot(122)
ax2.plot(data_sorted, p)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$p$')



test = np.zeros(5)




p_a = np.zeros((4, 2))
for i in range(4):
    p_a[i,0] = i
    p_a[i, 1] = 5-i
      
p_a[0, 1] = 0.3
p_a[1, 1] = 0.2
p_a[2, 1] = 0.4
p_a[3, 1] = 0.1

op_a = p_a[p_a[:,1].argsort()]

op_a[:,1] = np.cumsum(op_a[:,1])

print(op_a)
a = np.random.rand()
print("a: "+ str(a))
b = np.array(op_a[:,1] > a)
print("b: " + str(b))
c = op_a[b.argmax(),0]
print("c: " + str(c))

































