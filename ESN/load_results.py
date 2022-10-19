# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:11:27 2020

@author: Niki
"""

import pickle
import numpy as np


file_run1 = 'Run results\\criticality_run1.pkl'

file_bigtr_p1 = 'Run results\criticality_run_bigtrain_part1_output.pkl'
file_bigtr_p2 = 'Run results\criticality_run_bigtrain_part2_output.pkl'
file_bigtr_p3 = 'Run results\criticality_run_bigtrain_part3_output.pkl'
file_bigtr_p4 = 'Run results\criticality_run_bigtrain_part4_output.pkl'
file_bigtr_p5 = 'Run results\criticality_run_bigtrain_part5_output.pkl'



with open(file_run1, 'rb') as inp:
    run1 = pickle.load(inp)
    maxdet_run1 = np.max(run1['determinants'])

maxdet_bigtr = np.zeros((5, 2))
with open(file_bigtr_p1, 'rb') as inp:
    run_bigtr1 = pickle.load(inp)
    maxdet_bigtr[0,0] = np.max(run_bigtr1['determinants'])
    maxdet_bigtr[0,1] = np.argmax(run_bigtr1['determinants'])
with open(file_bigtr_p2, 'rb') as inp:
    run_bigtr2 = pickle.load(inp)
    maxdet_bigtr[1,0] = np.max(run_bigtr2['determinants'])
    maxdet_bigtr[1,1] = np.argmax(run_bigtr2['determinants'])
with open(file_bigtr_p3, 'rb') as inp:
    run_bigtr3 = pickle.load(inp)
    maxdet_bigtr[2,0] = np.max(run_bigtr3['determinants'])
    maxdet_bigtr[2,1] = np.argmax(run_bigtr3['determinants'])
with open(file_bigtr_p4, 'rb') as inp:
    run_bigtr4 = pickle.load(inp)
    maxdet_bigtr[3,0] = np.max(run_bigtr4['determinants'])
    maxdet_bigtr[3,1] = np.argmax(run_bigtr4['determinants'])
with open(file_bigtr_p5, 'rb') as inp:
    run_bigtr5 = pickle.load(inp)
    maxdet_bigtr[4,0] = np.max(run_bigtr5['determinants'])
    maxdet_bigtr[4,1] = np.argmax(run_bigtr5['determinants'])
    
    