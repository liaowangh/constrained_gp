# Sampling a truncated multivariate Gaussian by Metropolis-Hastings

# Author: Liaowang Huang <liahuang@student.ethz.ch>


import numpy as np
import scipy.linalg
from scipy.stats import multivariate_normal
from .RSM import RsmSampler


class MHSampler:
    def __init__(self, mu, Sigma, state, f, g, eta):
        """
        :param mu:    (m,) mean
        :param Sigma: (m,m) covariance matrix
        :param state: (m,) current state,
        :param f:     (q,m) matrix, where q is the number of linear constraints. The constraints require each component
                      of the m-dimensional vector fX+g to be non-negative
        :param g:     (q,) vector with the constant terms in the above linear constraints.
        :param eta:   scale factor
        """
        self.mu = mu
        self.Sigma = Sigma
        self.state = state
        self.f = f
        self.g = g
        self.eta = eta
        self.reject = 0

    def p(self, x):
        """
        Return the not normalized probability of a truncated multivariate normal at x

        :param x: (m,)
        """
        if self.f is not None and (not np.all(self.f@x + self.g >= 0)):
            return 0
        else:
            return np.exp(-0.5*(x - self.mu).dot(scipy.linalg.solve(self.Sigma, x - self.mu)))

    def q(self, y, x):
        """
        Return the probability of proposal distribution at y,
        the proposal distribution q(y|x)=N(x, eta*Sigma), where x is the current state

        :param y: (m,)
        :param x: (m,) the previous state
        """
        return multivariate_normal.pdf(y, mean=x, cov=self.eta*self.Sigma)

    def next_state(self):
        """
        sample a candidate state from the proposal distribution and
        then accept the state according to an appropriate criterion
        """
        old_state = self.state
        rsm_sampler = RsmSampler(old_state, self.eta * self.Sigma, self.f, self.g)
        new_state = rsm_sampler.rsm_tmg()
        r = np.minimum(1, self.p(new_state) / self.p(old_state))
        if np.random.uniform() >= r:
            new_state = old_state  # reject the candidate state
            self.reject += 1
        self.state = new_state
        return new_state


def mh(n, mu, Sigma, initial, f, g, eta, burn_in=30):
    """
    Sampling from a truncated multivariate normal Q(x)~TN(mu, Sigma, l, u),
    we use the Metropolis-Hastings method, a symmetric Gaussian proposal: q(x'|x) = N(x, eta*Sigma) is used.
    the proposed new state x' should sat

    :param n:       Number of samples.
    :param mu:      (m,) mean
    :param Sigma:   (m,m) covariance matrix
    :param initial: (m,) the starting state in sampling
    :param f:       (q,m) matrix, where q is the number of linear constraints. The constraints require each component
                    of the m-dimensional vector fX+g to be non-negative
    :param g:       (q,) vector with the constant terms in the above linear constraints.
    :param eta:     scale factor
    :param burn_in: The number of burn-in iterations. The Markov chain is sampled n + burn_in
                    times, and the last n samples are returned.
    """
    dim = len(mu)
    samples = np.zeros((n, dim))

    mhs = MHSampler(mu, Sigma, initial, f, g, eta)
    for i in range(burn_in):
        mhs.next_state()

    for i in range(n):
        samples[i] = mhs.next_state()
    # print("{} proposed states are rejected, while {} samples in total".format(mhs.reject, n+burn_in))
    return samples
