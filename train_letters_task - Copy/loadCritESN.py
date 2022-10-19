# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:16:50 2020

@author: Niki
"""

import pickle
import numpy as np

file1param = 'runData\\esn\\critical_esn_myenv_params.pkl'

with open(file1param, 'rb') as input:
    critpar1 = pickle.load(input)

file1out = 'runData\\esn\\critical_esn_myenv_output.pkl'

with open(file1out, 'rb') as input:
    critout1 = pickle.load(input)
    

file2param = 'runData\\esn\\critical_esn_myenv_params_run2.pkl'

with open(file1param, 'rb') as input:
    critpar2 = pickle.load(input)

file2out = 'runData\\esn\\critical_esn_myenv_output_run2.pkl'

with open(file2out, 'rb') as input:
    critout2 = pickle.load(input)
    
##
critout1['param_set_star']
dets1=critout1['dets']
np.max(dets1)
    
dets2=critout2['dets']
critout2['param_set_star']
np.max(dets2)