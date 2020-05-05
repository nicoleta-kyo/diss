# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:17:29 2020

@author: Niki
"""

i = np.tile(np.repeat(float('inf'),5),(3, 1))
i[0,0] = 3
i[0,1] = 2
i[1,:] = 5
i[2,0] = 4

i[i[:,0] != float('inf'),0]
i[i[:,1] != float('inf'),1]
np.nonzero(i)
