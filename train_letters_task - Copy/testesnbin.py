# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:54:28 2020

@author: Niki
"""

print('Instantiating network ' + str(irun))
                    
esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs,
             spectral_radius=spectral_radius, sparsity=sparsity,
             input_weights_scaling = input_weights_scaling,
             out_activation=out_activation, inverse_out_activation=inverse_out_activation,
             input_bias=input_bias)
print('RLS training...')
for ep in range(tr_inputs.shape[0]):
    epinputs = tr_inputs[ep,:,:]
    epoutputs = tr_outputs[ep,:,:]
    acts = esn.get_states(epinputs, extended=True, continuation=False)
    
    for actval,outval in zip(acts,epoutputs):
        outval = outval.reshape((1, -1))
        outval = esn.inverse_out_activation(outval.T)
        esn.RLSfilter.process_datum(actval.reshape(-1,1), outval.reshape(-1,1))               
print('Testing...')