# -*- coding: utf-8 -*-
"""
Implementation of RLS for uncentered data

"""

import numpy as np


class RLSFilterAnalyticIntercept():
    """
    Class representing the state of a recursive least squares estimator with
    intercept estimation.
    """

    def __init__(self, input_dim, output_dim, alpha=1.0, forgetting_factor=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.param_dim = input_dim

        self.alpha = alpha
        self.forgetting_factor = forgetting_factor

        self.t = 0.0

        self.intercept = np.zeros((self.output_dim, 1))

        self.theta = np.zeros((self.param_dim, self.output_dim))
        self.corrected_theta = np.zeros_like(self.theta)
        self.feat_mean = np.zeros((1, self.param_dim))
        self.output_mean = np.zeros((1, self.output_dim))
        self.covar = np.eye(self.param_dim) * alpha

    def _make_feature_vec(self, in_vec):
        assert(in_vec.shape == (self.input_dim, 1))

        return in_vec.transpose()

    def _update_covar(self, U, C, V):
        assert(U.shape == (self.param_dim, 2))
        assert(C.shape == (2, 2))
        assert(V.shape == (2, self.param_dim))

        inv_part = np.linalg.inv(C) + V.dot(self.covar).dot(U)
        update = self.covar.dot(U).dot(np.linalg.inv(inv_part)).dot(V).dot(self.covar)

        self.covar = (1.0 / self.forgetting_factor ** 2) * (self.covar - update)

    def _update_theta(self, C_t, feat, output):
        assert(feat.shape == (1, self.param_dim))
        assert(output.shape == (self.output_dim, 1))
        assert(C_t.shape == (self.param_dim, self.param_dim))

        inner_term = feat.transpose().dot(output.transpose()) - C_t.dot(self.theta)
        update = self.covar.dot(inner_term)
        self.theta = self.theta + update

    def _update_output_mean(self, output):
        assert(output.shape == (self.output_dim, 1))
        self.output_mean = (self.forgetting_factor * self.output_mean) + (1.0 / self.t) * (output.transpose() - (self.forgetting_factor * self.output_mean))

    def _update_feat_mean(self, feat):
        assert(feat.shape == (1, self.param_dim))
        self.feat_mean = (self.forgetting_factor * self.feat_mean) + (1.0 / self.t) * (feat - (self.forgetting_factor * self.feat_mean))

    def _make_U(self, feat):
        assert(feat.shape == (1, self.param_dim))
        return np.block([self.forgetting_factor * self.feat_mean.transpose(), feat.transpose()])

    def _make_V(self, feat):
        assert(feat.shape == (1, self.param_dim))
        return np.block([[self.forgetting_factor * self.feat_mean],[feat]])

    def _make_C(self):
        return (1 / ((self.forgetting_factor * self.t) ** 2)) * np.array([[((2.0 * self.t - 1.0) ** 2) - 2.0 * (self.t ** 2), -(2.0 * self.t - 1.0) * (self.t - 1.0)],
                         [-(2.0 * self.t - 1.0) * (self.t - 1.0), (self.t - 1.0) ** 2]])

    def process_datum(self, in_vec, output):
        feat = self._make_feature_vec(in_vec)
        self.t += 1.0

        if self.t == 1.0:
            self._update_feat_mean(feat)
            self._update_output_mean(output)
            return

        U = self._make_U(feat)
        V = self._make_V(feat)
        C = self._make_C()
        C_t = U.dot(C).dot(V)

        self._update_covar(U, C, V)
        self._update_output_mean(output)
        self._update_feat_mean(feat)
        self._update_theta(C_t, feat, output)
        self.corrected_theta = self.theta - ((2 * self.t - 1) * self.covar.dot(self.feat_mean.transpose()).dot(self.output_mean))
        self.intercept = (self.output_mean - self.feat_mean.dot(self.corrected_theta)).transpose()

    def get_identified_mats(self):
        """
        Returns the current estimated A, B, and c matrices.
        """
        return self.corrected_theta.transpose(), self.intercept

    def predict(self, in_vec):
        feat = self._make_feature_vec(in_vec)
        prediction = feat.dot(self.corrected_theta).transpose() + self.intercept
        assert(prediction.shape == (self.output_dim, 1))

        return prediction


        return prediction
    
 # ==============   
# 
#N = 100
#input_dim = 3
#output_dim = 2
#x = np.random.normal(0, 1, (N, input_dim)) # input matrix
#v = np.random.normal(0, 1, N) # noise
#d = np.zeros((N,output_dim))
#d[:,0] = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + v
#d[:,1] = 2.5*x[:,0] + 0.6*x[:,1] - 4.5*x[:,2] + v
#
#
#    
#filt = RLSFilterAnalyticIntercept(input_dim, output_dim)
#
#preds = np.zeros(d.shape)
#for i in range(N):
#    preds[i,:] = filt.predict(x[i,:].reshape(-1,1)).T
#    filt.process_datum(x[i,:].reshape(-1,1), d[i,:].reshape(-1,1))
#    
#    
#import matplotlib.pyplot as plt
#
#x = list(range(1,N+1))
#y = d[:,0]
#y2 = preds[:,0]
#
#fig, ax = plt.subplots()
#
## Using set_dashes() to modify dashing of an existing line
#line1, = ax.plot(x, y, label='actual')
#line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#
## Using plot(..., dashes=...) to set the dashing when creating a line
#line2, = ax.plot(x, y2, dashes=[6, 2], label='predicted')
#
#ax.legend()
#plt.show()
# 