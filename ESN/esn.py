
"""

ESN Implementation imported from: https://github.com/cknd/pyESN/blob/master/pyESN.py

Initialisation of reservoir weights and input weights done according to Jaeger's paper,
and implementation of augmented training algorithm following the paper:
    https://papers.nips.cc/paper/2318-adaptive-nonlinear-system-identification-with-echo-state-networks.pdf  

Readout training with Moore-Penrose Matrix Inverse or Ridge Regression (added)

"""

import numpy as np
from sklearn.linear_model import Ridge
import math
#import warnings
#warnings.filterwarnings("error")


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.
    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s
    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s

def atanh(x):
    
    atanhx = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            atanhx[i,j] = math.atanh(x[i,j] - np.finfo('float').eps)
        
    return atanhx

#def identity(x):
#    return x


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0.9,
                 noise=0.001,
                 readout='pseudo-inverse',
                 ridge_reg=None,
                 input_weights_scaling = 1,
                 input_scaling=None,input_shift=None,teacher_forcing=None, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=np.tanh, inverse_out_activation=np.arctanh,
                 silent=True, 
                 augmented=False,
                 transient=200):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            readout: type of readout 0 can be moonrose pseudo-inverse or ridge regression
            ridge_reg: regularisation value alpha if readout is Ridge
            
            input_weights_scaling: scaling of the input connection weights
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.                       
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
                        
            teacher_shift: additive term applied to the target signal
            teacher_scaling: factor applied to the target signal
            teacher_forcing: if True, feed the target back into output units

            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
    
            silent: supress messages
            augmented: if True, use augmented training algorithm
            transient: how many initial states to discard
            
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.readout = readout
        self.ridge_reg = ridge_reg
        
        self.input_weights_scaling = input_weights_scaling
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_shift = teacher_shift
        self.teacher_scaling = teacher_scaling
        self.teacher_forcing = teacher_forcing

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation

        self.silent = silent
        self.augmented = augmented
        self.transient = transient
        
        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        
#        # begin with a random matrix centered around zero:
#        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        
        # [nk] following Jaeger's paper:
        W = np.random.uniform(low = -1, high = 1, size = (self.n_reservoir, self.n_reservoir))
        
        # delete the fraction of connections given by (self.sparsity):
        W[np.random.RandomState().rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

#        # random input weights:
#        self.W_in = self.random_state_.rand(
#            self.n_reservoir, self.n_inputs) * 2 - 1
            
        # [nk] following Jaeger's paper:
        # added scaling
        self.W_in = np.random.uniform(low = -0.1, high = 0.1, size = (self.n_reservoir, self.n_inputs))*self.input_weights_scaling
                
        # random feedback (teacher forcing) weights:
        self.W_feedb = np.random.RandomState().rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern)
                             )
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))
            
        # [nk] add noise - the original code added noise after applying non-linearity!
        preactivation = preactivation + self.noise * (np.random.uniform(0,1,self.n_reservoir))
        
        return (np.tanh(preactivation))

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled
    
    def get_states(self, inputs, outputs, inspect=False, extended=True):
        """
        [nk]
        Collect the network's neuron activations.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's states for every input sample
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input, output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        augmented_states = np.zeros((inputs.shape[0], self.n_reservoir*2+2))
        # initialise first row with first input and zero states
        augmented_states[0,:] = np.hstack((np.hstack((inputs_scaled[0,:],np.zeros(self.n_reservoir))),
                                                    np.hstack((np.power(inputs_scaled[0,:],2),np.zeros(self.n_reservoir)))
                                                    ))
        
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])
            # x_squares(n) =  (u(n), x1(n), ... , xN(n), u^2(n), x1^2(n), ... , xN^2(n))
            augmented_states[n,:] = np.hstack((np.hstack((inputs_scaled[n,:],states[n,:])),
                                                    np.hstack((np.power(inputs_scaled[n,:],2),np.power(states[n,:],2)))
                                                    ))
        # include the raw inputs for states
        extended_states = np.hstack((inputs_scaled, states))
        
        # output states
        if self.augmented:
            out = augmented_states
        elif extended:
            out = extended_states
        else:
            out = states
       
        return out

    def fit(self, inputs, outputs, inspect):
        """
        [nk]
        Collect the network's reaction to training data, train readout weights.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform teacher signal:
        teachers_scaled = self._scale_teacher(outputs)
        
        # [nk] collect reservoir states
        states = self.get_states(inputs, outputs, inspect)

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # [nk] disregard the first few states:
#        transient = min(int(inputs.shape[1] / 10), 100)
        
        # Solve for W_out:
        if self.readout == 'pseudo-inverse':
            self.W_out = np.dot(np.linalg.pinv(states[self.transient:, :]),
                        self.inverse_out_activation(teachers_scaled[self.transient:, :])).T
        elif self.readout == 'ridge':
            self.readout = Ridge(alpha=self.ridge_reg)
            self.readout.fit(states[self.transient:, :], teachers_scaled[self.transient:, :])
        else:
            raise ValueError('Invalid readout parameter: Must be either "ridge" or "pseudo-inverse".')

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        # apply learned weights to the collected states:
        if not self.silent:
            print("training (squared mean squared) error:")
        if self.readout == 'pseudo-inverse': 
            pred_train = self._unscale_teacher(self.out_activation(
                    np.dot(states, self.W_out.T)))
        else:   #ridge
            pred_train = self._unscale_teacher(self.readout.predict(states))
        if not self.silent:
            print(np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train

    def predict(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.
        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
            augmented: if true, create a squared version of the network states
        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        
        if self.augmented:
            augmented_states = np.zeros((n_samples+1, self.n_reservoir*2+2))
            # initialise first row with augmented last input + last state
            augmented_states[0,:] = np.hstack((np.hstack((lastinput,laststate)),
                                                    np.hstack((np.power(lastinput,2),np.power(laststate,2)))
                                                    ))
            
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[
                n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            
            if self.augmented:
                augmented_states[n + 1,:] = np.hstack((np.hstack((inputs[n + 1,:],states[n + 1,:])),
                                                    np.hstack((np.power(inputs[n + 1,:],2),np.power(states[n + 1,:],2)))))
                
                if self.readout == 'pseudo-inverse':
                    outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,augmented_states[n + 1,:]))
                else:   # ridge
                    outputs[n + 1, :] = self.readout.predict(augmented_states[n + 1,:].reshape(1,-1))
            else:
                if self.readout == 'pseudo-inverse':
                    outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                                                           np.concatenate([inputs[n + 1, :], states[n + 1, :]])))
                else:  # ridge
                    outputs[n + 1, :] = self.readout.predict(np.concatenate([inputs[n + 1, :], states[n + 1, :]]).reshape(1,-1))   
        
        #unscaled_outputs = self._unscale_teacher(self.out_activation(outputs[1:]))
        # ! should not apply tanh function again!!
        unscaled_outputs = self._unscale_teacher(outputs[1:])
        
        return unscaled_outputs