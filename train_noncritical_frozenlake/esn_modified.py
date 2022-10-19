
"""

ESN Implementation imported from: https://github.com/cknd/pyESN/blob/master/pyESN.py

Initialisation of reservoir weights and input weights done according to Jaeger's paper,
and implementation of augmented training algorithm following the paper:
    https://papers.nips.cc/paper/2318-adaptive-nonlinear-system-identification-with-echo-state-networks.pdf  

Readout training with Moore-Penrose Matrix Inverse or Ridge Regression (added)

Changes for Otsuka's model architecture:
    - sigmoid function for update of memory layer (reservoir units): 
        gain=4, added bias for the reservoir activations
    - default out activation: identity
    - get_states method to work with continuation
    - fit and predict methods take reservoir units (output from get_states) directly
    - RLS regression for online learning
    
Intrinsic Motivation Schmidhuber
    - history object: for each n in N_T (episodes*num_steps): [state, action, state2, reward]
    - history is evaluated on the last <= 1000 time-steps
    - calculateReward methods

"""

import numpy as np
# from sklearn.linear_model import Ridge
from cannon_rlsfilter import RLSFilterAnalyticIntercept
import math
import warnings


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


# sigmoid
def sigmoid(x, gain=1): 
    return 1 / (1 + np.exp(-gain*x))

# !!! do I apply the gain to the predictions as well?
def inv_sigmoid(x, gain=1):
    return np.log( (x*gain) / (1 - (x*gain) ) )

def atanh(x):
    #x is of shape (1, teachers)
    
    atanhx = np.zeros(x.shape[1])
    for i,v in enumerate(x[0]):
        atanhx[i] = math.atanh(v)
        
    return atanhx
        
def identity(x):
    return x


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0,
                 noise=0.001,
                 readout='pseudo-inverse',
                 ridge_reg=None,
                 input_weights_scaling = 1,
                 input_scaling=None,input_shift=None,teacher_forcing=None, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=np.tanh, inverse_out_activation=atanh,
                 silent=True, 
                 augmented=False,
                 transient=200,
                 input_bias=0
                 ):
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
        self.n_outputs = n_outputs   # part will be obs, part will be reward
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
        
        self.input_bias = input_bias
        
        self.laststate = np.zeros(self.n_reservoir)
        self.lastextendedstate = np.zeros(self.n_reservoir+self.n_inputs)
        self.lastinput = np.zeros(self.n_inputs+self.input_bias)
        self.lastoutput = np.zeros(self.n_inputs)

        self.defaultHistEval = 1000    # look at last n time steps when evaluating history
        
        self.initweights()
        
    def initweights(self):
        
        # initialize recurrent weights:
        self.W = self.initialise_reservoir()
        
        # bias for the update of the states of the reservoir
        self.reservoir_bias = np.dot(np.repeat(-0.5,self.n_reservoir),self.W)
            
        # [nk] following Jaeger's paper:
        # added scaling
        self.W_in = np.random.uniform(low = -0.1, high = 0.1, size = (self.n_reservoir, self.n_inputs+self.input_bias))*self.input_weights_scaling
             
        # random feedback (teacher forcing) weights:
        self.W_feedb = np.random.RandomState().rand(
            self.n_reservoir, self.n_outputs) * 2 - 1
                
        # filter for online learning
        self.RLSfilter = RLSFilterAnalyticIntercept(self.n_reservoir+self.n_inputs+self.input_bias, self.n_outputs, forgetting_factor=0.995)
          
        
    def initialise_reservoir(self):
        
        # [nk] following Jaeger's paper:
        W = np.random.uniform(low = -1, high = 1, size = (self.n_reservoir, self.n_reservoir))
        # delete the fraction of connections given by (self.sparsity):
        W[np.random.RandomState().rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                W = W * (self.spectral_radius / radius)
            except:
                print("re-initialising reservoir because spectral radius was 0")
                self.initialise_reservoir()
              
        return W
    
    def resetState(self):
        self.laststate = np.zeros(self.n_reservoir)
        self.lastextendedstate = np.zeros(self.n_reservoir+self.n_inputs)
        self.lastinput = np.zeros(self.n_inputs)
        self.lastoutput = np.zeros(self.n_inputs)

    def _update(self, state, input_pattern, output_pattern=None):
        """performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        
#         pdb.set_trace()
        
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern)
                             )
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             )
            
        # [nk] add noise - the original code added noise after applying non-linearity!
        preactivation = preactivation + self.noise * (np.random.uniform(0,1,self.n_reservoir))
        
        # apply activation function to the reservoir with the necessary gain and the bias = -0.5*W
        activation = sigmoid(preactivation,4) + self.reservoir_bias
        
        return activation

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
    
    
    def get_states(self, inputs, extended, continuation, outputs=None, inspect=False):
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
        
        if outputs is not None and outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        n_samples = inputs.shape[0]
        
        # add bias to inputs if there is such
        if self.input_bias != 0:
            inputs = np.hstack((inputs, np.ones((n_samples,1))))
            
        # use last state, input, output
        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs+self.input_bias)
            lastoutput = np.zeros(self.n_outputs)
            
        if not self.silent:
            print("harvesting states...")    

        # create scaled input and output vector
        inputs_scaled = np.vstack([lastinput, self._scale_inputs(inputs)])
        if outputs is not None:
            teachers_scaled = np.vstack([lastoutput, self._scale_teacher(outputs)])
        # create states vector
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        
        
        if self.augmented:
            # create extended states vector
            lastaugmentedstate = np.hstack((np.hstack((lastinput, laststate)),
                                                        np.hstack((np.power(lastinput,2),np.power(laststate,2)))
                                                        ))
            augmented_states = np.vstack(
                    [lastaugmentedstate,np.zeros((n_samples, self.n_reservoir*2+2))])
            
        
        # activate the reservoir with the given input:
        for n in range(1, n_samples+1):
            if outputs is not None:
                states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])
            else:
                states[n, :] = self._update(states[n - 1], inputs_scaled[n, :])
            
            if self.augmented:
                # x_squares(n) =  (u(n), x1(n), ... , xN(n), u^2(n), x1^2(n), ... , xN^2(n))
                # ! teacher forcing version missing
                augmented_states[n,:] = np.hstack((np.hstack((inputs_scaled[n,:],states[n,:])),
                                                        np.hstack((np.power(inputs_scaled[n,:],2),np.power(states[n,:],2)))
                                                        ))
        # include the raw inputs for states
        extended_states = np.hstack((inputs_scaled, states))
        
        # remember the last state, input, output for later:
        self.laststate = states[-1, :]
        self.lastextendedstate = extended_states[-1,:]
        self.lastinput = inputs_scaled[-1, :]
        if outputs is not None:
            self.lastoutput = teachers_scaled[-1, :]
        
        # output states
        if self.augmented:
            out = augmented_states
        elif extended:
            out = extended_states
        else:
            out = states
       
        return out[1:]    #output without last state

    def fit(self, outputs, inputs, continuation, inspect=False):
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
        states = self.get_states(inputs, extended=True, continuation=continuation)

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        
        # Solve for W_out:
        if self.readout == 'pseudo-inverse':
            self.W_out = np.dot(np.linalg.pinv(states[self.transient:, :]),
                        self.inverse_out_activation(teachers_scaled[self.transient:, :])).T
        elif self.readout == 'ridge':
            self.readout = Ridge(alpha=self.ridge_reg)
            self.readout.fit(states[self.transient:, :], teachers_scaled[self.transient:, :])
        else:
            raise ValueError('Invalid readout parameter: Must be either "ridge" or "pseudo-inverse".')

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

    
    def predict(self, states):
        """
        Apply the learned weights to the network's reactions to new input.
        Args:
            states: the reservoir of the network which has been activated by the input
        Returns:
            Array of output activations
        """
        n_samples = states.shape[0]
          
        # output predictions for each input
        outputs = np.zeros((n_samples, self.n_outputs))
        for n in range(n_samples):
            if self.readout == 'pseudo-inverse':
                outputs[n, :] = self.out_activation(np.dot(self.W_out,states[n,:]))
            else:   # ridge
                outputs[n, :] = self.readout.predict(states[n,:].reshape(1,-1))
        
        unscaled_outputs = self._unscale_teacher(outputs)
        
        return unscaled_outputs 
        
    
    # array to store all obs + rewards
    def setHistory(self, episodes, steps):

        self.num_elem_hist = self.n_inputs + self.n_outputs
        self.history = np.repeat(np.inf, episodes*steps*self.num_elem_hist).reshape(episodes, steps, self.num_elem_hist)
    
    # internal reward: evaluate the current network on the history, fit it to the last teacher output from the history,
    # evaluate the new network, return difference between the two
    # Schmidhuber --> int_reward = C(p_old, hist) - C(p_new, hist)
    def calculateInternalReward(self, allEpisodes=False):
        
#         pdb.set_trace()
            
        #------ calc C(p_old, hist)
        hist = self.history[self.history != float("inf")]  # take all time steps that happened
        hist = hist.reshape(int(len(hist)/self.num_elem_hist),self.num_elem_hist) # reshape into (all time steps, hist elements)
        
        # take all history or last 10k times steps
        if allEpisodes or hist.shape[0] <= self.defaultHistEval:
            inputs = hist[:,:self.n_inputs]
            teachers = hist[:,self.n_inputs:]
        else:
            inputs = hist[-self.defaultHistEval:,:self.n_inputs]
            teachers = hist[-self.defaultHistEval:,self.n_inputs:]
        
        # apply inverse ou activation to the teacher signal
        teachers = self.inverse_out_activation(teachers)
        
        # get reservoir activations for all history
        res_states = self.get_states(inputs, extended=True, continuation=False)  #continuation is False because starts from first state
        
        # get predictions by applying the rls filter
        preds1 = np.zeros((inputs.shape[0], self.n_outputs), dtype='int64')
        for i in range(inputs.shape[0]):
            preds1[i,:] = self.out_activation(self.RLSfilter.predict(res_states[i,:].reshape(-1,1))).T
        
        # getting a different runtime warning so i just changed episodes to be evaluated to 1000
        # calculate predictor 
        quality1 = np.mean(np.abs((preds1 - teachers).astype('int64')))
        
        
        #--------- update filter with last input-output
        self.RLSfilter.process_datum(res_states[-1,:].reshape(-1,1), teachers[-1,:].reshape(-1,1))
        
        #------- calc C(p_new, hist)
        preds2 = np.zeros((inputs.shape[0], self.n_outputs), dtype='int64')
        for i in range(inputs.shape[0]):
            preds2[i,:] = self.out_activation(self.RLSfilter.predict(res_states[i,:].reshape(-1,1)).T)  
    
        # calculate predictor quality and save it
        self.quality = round(np.mean(np.abs((preds2 - teachers).astype('int64'))),2)

        rew = quality1 - self.quality
        
        posr = rew if rew > 0 else 0  # disregard negative reward
        bposr = np.tanh(posr)  # bound
        
        return bposr, self.quality
    
        
