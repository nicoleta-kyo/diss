# -*- coding: utf-8 -*-


"""The classical non-differentiable Friedman-Rafsky test. """

from scipy.sparse.csgraph import minimum_spanning_tree as mst
from torch.autograd import Function
import numpy as np
import torch

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


class MSTFn(Function):
    """Compute the minimum spanning tree given a matrix of pairwise weights."""
    @staticmethod
    def forward(ctx, weights):
        """Compute the MST given the edge weights.
        The behaviour is the same as that of ``minimum_spanning_tree` in
        ``scipy.sparse.csgraph``, namely i) the edges are assumed non-negative,
        ii) if ``weights[i, j]`` and ``weights[j, i]`` are both non-negative,
        their minimum is taken as the edge weight.
        Arguments
        ---------
        weights: :class:`torch:torch.Tensor`
            The adjacency matrix of size ``(n, n)``.
        Returns
        -------
        :class:`torch:torch.Tensor`
            An ``(n, n)`` matrix adjacency matrix of the minimum spanning tree.
            Indices corresponding to the edges in the MST are set to one, rest
            are set to zero.
            If both weights[i, j] and weights[j, i] are non-zero, then the one
            will be located in whichever holds the *smaller* value (ties broken
            arbitrarily).
        """
        mst_matrix = mst(weights.cpu().numpy()).toarray() > 0
        assert int(mst_matrix.sum()) + 1 == weights.size(0)
        return torch.Tensor(mst_matrix.astype(float))
    
class FRStatisticDiffSample(object):
    """The classical Friedman-Rafsky test :cite:`friedman1979multivariate`.
    Arguments
    ----------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample."""
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed Friedman-Rafsky test statistic.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.FRStatistic.pval`.
        Returns
        -------
        float
            The number of edges that do connect points from ONE SAMPLE TO THE OTHER.
        """
        n_1 = sample_1.size(0)
        assert n_1 == self.n_1 and sample_2.size(0) == self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        mstfn = MSTFn.apply
        mst_matrix = mstfn(diffs)

        statistic = mst_matrix[:n_1, n_1:].sum() + mst_matrix[n_1:, :n_1].sum()

        if ret_matrix:
            return statistic, mst_matrix
        else:
            return statistic
        
class FRStatisticSameSample(object):
    """The classical Friedman-Rafsky test :cite:`friedman1979multivariate`.
    Arguments
    ----------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample."""
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed Friedman-Rafsky test statistic.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.FRStatistic.pval`.
        Returns
        -------
        float
            The number of edges that do connect points from the *same* sample.
        """
        n_1 = sample_1.size(0)
        assert n_1 == self.n_1 and sample_2.size(0) == self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        mst_matrix = MSTFn()(diffs)

        statistic = mst_matrix[:n_1, :n_1].sum() + mst_matrix[n_1:, n_1:].sum()

        if ret_matrix:
            return statistic, mst_matrix
        else:
            return statistic
