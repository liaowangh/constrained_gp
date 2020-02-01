# Sampling a truncated multivariate Gaussian by Rejection sampling from Mode
# ("A New Rejection Sampling Method for Truncated Multivariate Gaussian Random Variables
#   Restricted to Convex Sets" https://hal.archives-ouvertes.fr/hal-01063978/document)

# Author: Liaowang Huang <liahuang@student.ethz.ch>

import numpy as np
import cvxpy as cp


class RsmSampler:
    def __init__(self, mu, Sigma, f, g):
        """
        :param mu:    (m,) mean
        :param Sigma: (m,m) covariance matrix
        :param f:     (q,m) matrix, where q is the number of linear constraints. The constraints require each component
                      of the m-dimensional vector fX+g to be non-negative
        :param g:     (q,) vector with the constant terms in the above linear constraints.
        """
        self.mu = mu
        self.Sigma = Sigma
        if f is not None:
            valid = np.logical_and(g < np.inf, g > -np.inf)
            g = g[valid]
            f = f[valid]
            self.mode = mu
        self.f = f
        self.g = g
        self.reject = 0
        self.mode = self.mode_solver()

    def mode_solver(self):
        """
        mode = arg min x^T*Sigma^-1*x
        """
        if self.f is None:
            return self.mu
        m = len(self.mu)
        xi = cp.Variable(m)
        obj = cp.Minimize(cp.matrix_frac(xi, self.Sigma))

        constraints = [self.f * (xi + self.mu) + self.g >= 0]

        prob = cp.Problem(obj, constraints)
        prob.solve()
        # print("status:", prob.status)
        if prob.status != "optimal":
            raise ValueError('cannot compute the mode')
        return xi.value

    def rsm_tmg(self):
        """
        Sampling from a multivariate normal N(mu, Sigma) with constraints f*x+g >= 0
        the rejection sampling method in paper
        A New Rejection Sampling Method for Truncated Multivariate Gaussian Random Variables Restricted to Convex Set.
        is used.
        """
        if self.f is None:
            return np.random.multivariate_normal(self.mu, self.Sigma)

        while True:
            state = np.random.multivariate_normal(self.mode, self.Sigma)
            u = np.random.uniform()
            if np.all(self.f @ (state + self.mu) + self.g >= 0) and \
                    u <= np.exp(self.mode.dot(np.linalg.solve(self.Sigma, self.mode)) -
                                state.dot(np.linalg.solve(self.Sigma, self.mode))):
                break
            self.reject += 1
        return state + self.mu


def rsm(n, mu, Sigma, f, g, verbose=False):
    """
    Sampling from a multivariate normal N(mu, Sigma) with constraints f*x+g >= 0
    the rejection sampling method is used.

    :param n:     Number of samples.
    :param mu:    (m,) mean
    :param Sigma: (m,m) covariance matrix.
    :param f:     (q,m), f*x+g >= 0 must be satisfied.
    :param g:     (q,)
    :param verbose: print acceptance rate if true.
    """

    rsm_sampler = RsmSampler(mu, Sigma, f, g)
    dim = len(mu)
    samples = np.zeros((n, dim))

    for i in range(n):
        samples[i] = rsm_sampler.rsm_tmg()

    if verbose:
        print("Acceptance rate is {}".format(n / (n + rsm_sampler.reject)))
    return samples
