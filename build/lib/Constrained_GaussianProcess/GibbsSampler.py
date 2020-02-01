# Gibbs sampling for truncated multivariate normal X~N(0,I) with constraints fX+g>=0
# suppose X=[X1,...Xn]', f=[f1,...fn]
# Then Xj|X-j ~ N(0,1) with constraints fj*Xj + f-j*X-j + g >= 0

# Author: Liaowang Huang <liahuang@student.ethz.ch>

import numpy as np


def truncated_gaussian(f, g):
    """
    sample from X~N(0,I) with constrained f*x+g >= 0
    """
    x = np.random.normal()
    while not np.all(f*x+g >= 0):
        x = np.random.normal()
    return x


class GibbsSampler:
    def __init__(self, dim, init, f, g):
        """
        :param dim:  dimension
        :param init: (dim, ), the initial value
        :param f:    (q, dim), coefficient for linear constraints
        :param g:    (q,), linear constraints: f*X+g >= 0
        """
        self.dim = dim
        self.state = init
        if f is not None:
            valid = np.logical_and(g < np.inf, g > -np.inf)
            g = g[valid]
            f = f[valid]
        self.f = f
        self.g = g

    def bound_j(self, j):
        """
        return fj and f-j*X-j+g
        """
        if self.f is None:
            return None, None
        else:
            fj = self.f[:, j]
            f_j = np.delete(self.f, j, 1)
            X_j = np.delete(self.state, j, 0)
            g_hat = self.g + f_j@X_j  # g_hat = g+f-j*X-j, so fj*Xj + g_hat >= 0
            return fj, g_hat
        # l = -np.inf
        # u = np.inf
        # if self.f is not None:
        #     fj = self.f[:, self.j]
        #     f_j = np.delete(self.f, self.j, 1)
        #     X_j = np.delete(self.state, self.j, 0)
        #     g_hat = -self.g - f_j@X_j  # g_hat = -g-f-j*X-j, so fj*Xj >= g_hat
        #     pos_f = fj > 0
        #
        #     if not np.all(pos_f):
        #         # if j are all positive, the upper bound is np.inf
        #         u = np.min(g_hat[~pos_f] / fj[~pos_f])
        #     if not np.all(~pos_f):
        #         # fj are all negative, the lower bound is -np.inf
        #         l = np.max(g_hat[pos_f] / fj[pos_f])
        # return l, u

    def sample_next(self):
        """
        generate next state
        """
        for j in range(self.dim):
            if self.f is None:
                x = np.random.normal()
            else:
                f_hat, g_hat = self.bound_j(j)
                x = truncated_gaussian(f_hat, g_hat)
            self.state[j] = x
        return self.state


def gibbs(n, mu, M, initial, f=None, g=None, burn_in=30):
    """
    This function generates samples from a Markov chain whose equilibrium distribution is a d-dimensional
    multivariate Gaussian truncated by linear inequalities. The probability log density is
    log p(X) = -0.5 (X-mu)^T M^-1 (X-mu) + const
    in terms of a covariance matrix M and a mean vector mu. The constraints are imposed as explained below.
    Gibbs sampling method is used.

    :param n:       Number of samples.
    :param mu:      (m,) vector for the mean of multivariate Gaussian density
    :param M:       (m,m) covariance matrix of the multivariate Gaussian density
    :param initial: (m,) vector with the initial value of the Markov chain. Must satisfy
                    the truncation inequalities strictly.
    :param f:       (q,m) matrix, where q is the number of linear constraints. The constraints require each component
                    of the m-dimensional vector fX+g to be non-negative
    :param g:       (q,) vector with the constant terms in the above linear constraints.
    :param burn_in: The number of burn-in iterations. The Markov chain is sampled n + burn_in
                    times, and the last n samples are returned.

    :return: (n, m)
    """

    dim = len(mu)
    if M.shape[1] != dim:
        raise ValueError("The covariance matrix must be square.")

    if len(initial) != dim:
        raise ValueError("Wrong length for initial value vector.")

    # verify that M is positive definite, it will raise an error if M is not SPD
    R = np.linalg.cholesky(M)  # R@R.T = M

    # we change variable to the canonical frame, and transform back after sampling
    # X ~ N(mu, M), then R^-1(X-mu) ~ N(0, I)
    init_trans = np.linalg.solve(R, initial - mu)  # the new initial value

    if f is not None:
        if f.shape[0] != len(g) or f.shape[1] != dim:
            raise ValueError("Inconsistent linear constraints. f must \
                              be an d-by-m matrix and g an d-dimensional vector.")
        # g may contains infinity, extract valid constraints
        valid = np.logical_and(g < np.inf, g > -np.inf)
        g = g[valid]
        f = f[valid]

        # verify initial value satisfies linear constraints
        if np.any(f@initial+g < 0):
            raise ValueError("Initial point violates linear constraints.")

        # map linear constraints to canonical frame
        f_trans = f@R
        g_trans = f@mu+g

        sampler = GibbsSampler(dim, init_trans, f_trans, g_trans)
    else:
        sampler = GibbsSampler(dim, init_trans, f, g)

    samples = np.zeros((n, dim))
    for i in range(burn_in):
        sampler.sample_next()
    for i in range(n):
        # for j in range(dim):
        #     sampler.sample_next()
        samples[i] = sampler.sample_next()

    # transform back
    return samples @ R.T + mu
