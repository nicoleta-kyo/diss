# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:11:27 2020

@author: Niki
"""

import pickle
import numpy as np


file_run1 = 'Run results\\criticality_run1.pkl'

file_bigiter_p3 = 'Run results\criticality_run_bigiter_part2_output3.pkl'
file_bigiter_p1 = 'Run results\criticality_run_bigtrain_part1_output.pkl'
#file_bigtr_p3 = 'Run results\criticality_run_bigtrain_part3_output.pkl'
#file_bigtr_p4 = 'Run results\criticality_run_bigtrain_part4_output.pkl'
#file_bigtr_p5 = 'Run results\criticality_run_bigtrain_part5_output.pkl'



with open(file_run1, 'rb') as inp:
    run1 = pickle.load(inp)
    maxdet_run1 = np.max(run1['dets'])


maxdet_bigiter = np.zeros((4, 2))
with open(file_bigiter_p3, 'rb') as inp:
    run_bigiter3 = pickle.load(inp)
    maxdet_bigiter[0,0] = np.max(run_bigiter3['dets'])
    maxdet_bigiter[0,1] = np.argmax(run_bigiter3['dets'])
    
with open(file_bigiter_p1, 'rb') as inp:
    run_bigiter1 = pickle.load(inp)
    maxdet_bigiter[1,0] = np.max(run_bigiter1['dets'])
    maxdet_bigiter[1,1] = np.argmax(run_bigiter1['dets'])
    
with open(file_bigtr_p3, 'rb') as inp:
    run_bigtr3 = pickle.load(inp)
    maxdet_bigtr[2,0] = np.max(run_bigtr3['dets'])
    maxdet_bigtr[2,1] = np.argmax(run_bigtr3['dets'])
with open(file_bigtr_p4, 'rb') as inp:
    run_bigtr4 = pickle.load(inp)
    maxdet_bigtr[3,0] = np.max(run_bigtr4['dets'])
    maxdet_bigtr[3,1] = np.argmax(run_bigtr4['dets'])
with open(file_bigtr_p5, 'rb') as inp:
    run_bigtr5 = pickle.load(inp)
    maxdet_bigtr[4,0] = np.max(run_bigtr5['dets'])
    maxdet_bigtr[4,1] = np.argmax(run_bigtr5['dets'])
    
# get params
part2parsf = 'Run results\criticality_run_bigtrain_part2_params.pkl'
with open(part2parsf, 'rb') as inp:
    part2pars = pickle.load(inp)