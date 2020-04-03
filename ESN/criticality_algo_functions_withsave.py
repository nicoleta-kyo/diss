# -*- coding: utf-8 -*-
"""
Implementation of "Determination of the edge of criticality in echo state
networks through Fisher information maximization"

"""

import numpy as np
from esn import ESN
import torch
#from torch_two_sample import FRStatistic
from FRTest import FRStatisticDiffSample  #import modified FRStatistic function!!
from numpy import matrix, linalg
import matlab
import matlab.engine as mateng
from itertools import combinations
import pickle
#from ttictoc import TicToc



""""
Create duplication matrix which transforms a half-vector representation
of a matrix into a vector representation
Params: 
   dims      the number of the diagonal elements of the matrix
Returns:
   dup_mat   the duplication matrix of type np.matrix
"""
def create_dup_mat(dims):
    D = np.eye(dims**2)
    gamma = []
    delta = []      
    for i in range(1,dims):
        gamma = gamma + [i+dims*j for j in range(i,dims)]
    for i in range(1,dims):
        delta = delta + [i+dims*(i-1)+j for j in range(1,dims-i+1)]
    for i in range(int(dims*(dims-1)/2)):
        D[:,delta[i]-1] = D[:,delta[i]-1] + D[:,gamma[i]-1]
    dup_mat = np.empty((D.shape[0],1))
    #gamma_1 = np.array(gamma)-1
    for i in range(D.shape[1]):
        if i not in np.array(gamma)-1:
            dup_mat = np.hstack((dup_mat, D[:,i].reshape((D.shape[0], 1))))   
    dup_mat = matrix(dup_mat[:,1:]) 
    return dup_mat

""""
Create shuffling matrix which reorders the elements of the half-vector object from
the diagonal elements being first to column-wise order
Params: 
   d      the number of the diagonal elements of the matrix
Returns:
   S      the shuffling matrix of type np.matrix
"""
def create_shuff_mat(d):
    S = np.zeros((int(d*(d+1)/2),int(d*(d+1)/2)))
    I = np.eye(int(d*(d+1)/2))
    gamma = [1]
    for i in range(2,d+1):
        gamma = gamma + [int(1+d*(i-1)-(i-1)*(i-2)/2)]
    for i in range(d):
        S[:,i] = I[:,gamma[i]-1]
    newI = np.empty((I.shape[0],1))
    for i in range(I.shape[1]):
        if i not in np.array(gamma)-1:
            newI = np.hstack((newI, I[:,i].reshape((I.shape[0], 1))))   
    newI = newI[:,1:]
    for i,j in zip(list(range(d,int(d*(d+1)/2))),list(range(newI.shape[1]))):
        S[:,i] = newI[:,j]
    return matrix(S)


""""
Use duplication and shuffling matrix to create full vector from half vector
Params: 
   Fhvec   matrix d(d+1)/2 by 1
   d       dimensions in the hyperparameter space
Returns:
    Fvec   matrix d**2 by 1
"""
def Fhvec_Fvec(d, Fhvec, rev):
    D = create_dup_mat(d)
    S = create_shuff_mat(d)
    Fvec = D*(S*Fhvec)
    return Fvec

""""
Do reverse operation of the method Fhvec_Fvec
Params: 
   Fvec   matrix d*d by 1
   d       dimensions in the hyperparameter space
Returns:
    Fvec   matrix d(d+1)/2 by 1
"""
def Fvec_Fhvec(d, Fvec):
    D = create_dup_mat(d)
    S = create_shuff_mat(d)
    Fhvec = S.I*D.I*Fvec
    return Fhvec

"""
Calculate the divergence between the states of the network configured with the original set of params
and the states of the network configured with the perturbed params using the Friedman-Rafsky test statistic
Params: 
    states1     activations of the original network
    states2     activations of the perturbed network
Returns:
    div         the divergence
"""
def calculate_divergence(states1, states2):
    s1 = torch.from_numpy(states1)
    s2 = torch.from_numpy(states2)
        
    fr_test = FRStatisticDiffSample(s1.size(0), s2.size(0))
    stat = fr_test(s1, s2)
    
    div = 1 - stat.numpy()*(states1.shape[0]+states2.shape[0])/(2*states1.shape[0]*states2.shape[0])  
    
    return div
###


# create esn object with given parameters(first set is with non-optimised params, second is params to be optimised)
def create_esn(config, config_to_opt):
    esn = ESN(n_inputs=config['n_inputs'] ,
                    n_outputs=config['n_outputs'],
                    n_reservoir=config['n_reservoir'],
                    augmented = config['augmented'],
                     transient = config['transient'],
                     # to optimise - not great that the order of the params is hardcoded
                    spectral_radius=config_to_opt[0],
                    sparsity=config_to_opt[1],
                     input_weights_scaling=config_to_opt[2],
                     
                     )
    return esn

def compute_fim(eng, d, hatF_hvec, R, v_theta):
    
    D = create_dup_mat(d)   # create shuffling and duplication matrices needed for F - Fhvec transformations
    S = create_shuff_mat(d)

    md=float(d)     # d needs to be double in matrix (mapped to float in python)
    #convert variables to matlab arrays
    mS = matlab.double([x.tolist() for x in np.array(S)])
    mD = matlab.double([x.tolist() for x in np.array(D)])
    mhatF_hvec = matlab.double([x.tolist() for x in np.array(hatF_hvec)])
    mR = matlab.double([x.tolist() for x in np.array(R)])
    mv_theta = matlab.double([x.tolist() for x in np.array(v_theta)])
    
    # run matlab function
    print("Running cvx_opt in Matlab...")
    mat, cvx_status = eng.cvx_opt_fim(md,mS,mD,mhatF_hvec, mR, mv_theta, nargout=2)
        
    # retrieve matrix if cvx opt is solved, otherwise return zero matrix
    fim = matrix(np.zeros((d,d)))
    if cvx_status == 'Solved':
        fim = matrix(mat)
        
    return fim
    
def check_bounds(set):
    # not great, hardcoded for the specific parameters
    
    #spectral radius
    set[0] = 0.1 if set[0] < 0.1 else set[0]
    #resevoir sparsity
    set[1] = 0.1 if set[1] < 0.1 else set[1]
    set[1] = 0.98 if set[1] > 0.98 else set[1]
    #input weights scaling
    set[2] = 0.1 if set[2] < 0.1 else set[2]
    set[2] = 1 if set[2] > 1 else set[2]    # not sure if that is necessary?
    
    return set
        



#============================== Algorithm 1 from the paper

def det_criticality(esn_config, narma_in, narma_out, param_space, num_iter, num_trials, num_pert, sigma_p, filenmoutput):
    
    out_dict = {}  # dict to save output

    print("Opening Matlab engine...")
    eng = mateng.start_matlab()      # open matlab engine
    
    d = param_space.shape[1]
    
    param_set = param_space[0,:].reshape(d,1)
    param_set_star = param_set                   # select initial param config
    
    det = float("-inf")     # initialise the FIM determinant
    
    dets = np.zeros(num_iter)    
    for it in range(num_iter):        #different stopping criterion?? for now i'd go through all parameter sets
      
#        #create time object
#        timer = TicToc()
#        timer.tic()
        
        fims = np.zeros((num_trials, d, d))
        # !!!! REMOVE LATER: to count unsolved fims
        unsolved_count = 0     
        for t in range(num_trials):
            esn = create_esn(esn_config, param_set)
            states = esn.get_states(narma_in, narma_out, extended = False)
        #        states = states[1:states.shape[0],:]    # get rid of first zero state
            
            R = matrix(np.zeros(( num_pert, int(d*(d+1)/2)))) # matrix with perturbation vectors as rows (**specified in the reference)
            D_a = np.zeros(num_pert)   # matrix with divergence values for every perturbation vector
            for j in range(num_pert):
                r = np.random.normal(0, sigma_p, d)  # create perturbation vector
                
                # create perturbation matrix (U) as defined in https://arxiv.org/pdf/1408.1182.pdf
                pairs = list(combinations(list(range(len(r))), 2))
                R[j,:] = list(np.power(r,2)) + [2*r[m]*r[n] for m,n in pairs]
                
                p_param_set = param_set + r.reshape(d,1)     # generate perturbed parameter space
                p_param_set = check_bounds(p_param_set)     #check if perturbed parameters are valid and change if not
                
                esn = create_esn(esn_config, p_param_set)   # create perturbed esn
                st = esn.get_states(narma_in, narma_out, extended=False)
        #            st = st[1:st.shape[0],:]    # get rid of first zero 
        
                print("Computing FR statistic " + str(j+1) + "..." + " from trial " + str(t+1) + "..." + "iteration " + str(it) + "...")
                D_a[j] = calculate_divergence(states, st)   # calculate the divergence between the original and the perturbed states by using the Friedman-Rafsky test
            
            # create vector with divergences
            v_theta = matrix(np.array([2*D_a[j] for j in range(len(D_a))]).reshape(len(D_a),1))
            
            hatF_hvec = (R.T*R).I*R.T*v_theta   #least sq
            
            try: 
                F = compute_fim(eng, d, hatF_hvec, R, v_theta)   #compute FIM
            except:                                 # reopen engine if it timed out, not great cause it handles any error
                eng = mateng.start_matlab()
                F = compute_fim(eng, d, hatF_hvec, R, v_theta)
            
            # check if fim is 0, !! dumb way but best i can do now
            # !!!! REMOVE LATER:
            if not np.any(F):
                unsolved_count = unsolved_count+1
                
            fims[t,:,:] = F
           
        fims = np.ma.masked_equal(fims, 0)
        av_fim = np.average(fims, (0))     # computer average fim over the trials
        dets[it] = linalg.det(av_fim)     # store new determinant
        if dets[it] > det:                # if new determinant > curr best, update curr best
            det = dets[it]
            param_set_star = param_set
        
        # rewrite output file every ten iterations
        if (it+1) % 2 == 0:
            out_dict['dets'] = dets
            out_dict['param_set_star'] = param_set_star
            
            with open(filenmoutput, 'wb') as output:
                pickle.dump(out_dict, output, pickle.HIGHEST_PROTOCOL)
        
        if (it+1) < param_space.shape[0]:
            param_set = param_space[it+1,:].reshape(d,1)      # choose next param set to evaluate
        
#        timer.toc()
#        print(timer.elapsed)
      

#    return (dets, param_set_star)



    
    
    

        



    
