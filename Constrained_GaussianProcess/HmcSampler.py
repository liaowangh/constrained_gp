# Python implementation of "Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussian"
# this script is written based on https://cran.r-project.org/web/packages/tmg/index.html

# Author: Liaowang Huang <liahuang@student.ethz.ch>

import numpy as np
import scipy.linalg


class HmcSampler:
    min_t = 0.00001

    def __init__(self, dim, init, f, g, verbose):
        """

        :param dim:  dimension
        :param init: (dim, ), the initial value for HMC
        :param f:    (q, dim), coefficient for linear constraints
        :param g:    (q,), linear constraints: f*X+g >= 0
        """
        self.dim = dim
        self.lastSample = init
        self.f = f
        self.g = g
        self.verbose = verbose

    def getNextLinearHitTime(self, a, b):
        """
        the position x(t) = a * sin(t) + b * cos(t)

        :param a: (dim, ) initial value for a (initial velocity)
        :param b: (dim, ) initial value for b (initial position)
        :return: hit_time: the time for the hit
                 cn : the cn-th constraint is active at hit time.
        """
        hit_time = 0
        cn = 0

        if self.f is None:
            return hit_time, cn

        f = self.f
        g = self.g
        for i in range(f.shape[0]):
            # constraints: f[i].dot(x)+g[i] >= 0
            fa = f[i].dot(a)
            fb = f[i].dot(b)
            u = np.sqrt(fa*fa + fb*fb)
            # if u > g[i] and u > -g[i]:
            if -u < g[i] < u:
                # otherwise the constrain will always be satisfied
                phi = np.arctan2(-fa, fb)  # -pi < phi < pi
                t1 = np.arccos(-g[i]/u) - phi  # -pi < t1 < 2*pi

                if t1 < 0:
                    t1 += 2 * np.pi  # 0 < t1 < 2*pi
                if np.abs(t1) < self.min_t or \
                   np.abs(t1-2*np.pi) < self.min_t:
                    t1 = 0

                t2 = -t1 - 2*phi  # -4*pi < t2 < 2*pi
                if t2 < 0:
                    t2 += 2*np.pi  # -2*pi < t2 < 2*pi
                if t2 < 0:
                    t2 += 2*np.pi  # 0 < t2 < 2*pi

                if np.abs(t2) < self.min_t or \
                   np.abs(t2 - 2 * np.pi) < self.min_t:
                    t2 = 0

                if t1 == 0:
                    t = t2
                elif t2 == 0:
                    t = t1
                else:
                    t = np.minimum(t1, t2)

                if self.min_t < t and (hit_time == 0 or t < hit_time):
                    hit_time = t
                    cn = i
        return hit_time, cn

    def verifyConstraints(self, b):
        """

        :param b:
        :return:
        """
        if self.f is not None:
            return np.min(self.f@b + self.g)
        else:
            return 1

    def sampleNext(self):
        T = np.pi / 2  # how much time to move
        b = self.lastSample
        dim = self.dim

        count_sample_vel = 0

        while True:
            velsign = 0
            # sample new initial velocity
            a = np.random.normal(0, 1, dim)

            count_sample_vel += 1
            if self.verbose and count_sample_vel % 50 == 0:
                print("Has sampled %d times of initial velocity." % count_sample_vel)

            tt = T  # the time left to move
            while True:
                t, c1 = self.getNextLinearHitTime(a, b)
                # t: how much time to move to hit the boundary, if t == 0, move tt
                # c1: the strict constraint at hit time

                if t == 0 or tt < t:
                    # if no wall to be hit (t == 0) or not enough
                    # time left to hit the wall (tt < t)
                    break

                tt -= t  # time left to move after hitting the wall
                new_b = np.sin(t) * a + np.cos(t) * b  # hit location
                hit_vel = np.cos(t) * a - np.sin(t) * b  # hit velocity
                b = new_b
                # reflect the velocity and verify that it points in the right direction
                f2 = np.dot(self.f[c1], self.f[c1])
                alpha = np.dot(self.f[c1], hit_vel) / f2
                a = hit_vel - 2*alpha*self.f[c1]  # reflected velocity

                velsign = a.dot(self.f[c1])

                if velsign < 0:
                    # get out of inner while, resample the velocity and start again
                    # this occurs rarelly, due to numerical instabilities
                    break

            if velsign < 0:
                # go to the beginning of outer while
                continue

            bb = np.sin(tt) * a + np.cos(tt) * b
            check = self.verifyConstraints(bb)
            if check >= 0:
                # verify that we don't violate the constraints
                # due to a numerical instability
                if self.verbose:
                    print("total number of velocity samples : %d" % count_sample_vel)

                self.lastSample = bb
                return bb


def tmg(n, mu, M, initial, f=None, g=None, burn_in=30, verbose=False):
    """
    This function generates samples from a Markov chain whose equilibrium distribution is a d-dimensional
    multivariate Gaussian truncated by linear inequalities. The probability log density is
    log p(X) = -0.5 (X-mu)^T M^-1 (X-mu) + const
    in terms of a covariance matrix M and a mean vector mu. The constraints are imposed as explained below.
    The Markov chain is built using the Hamiltonian Monte Carlo technique.

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
    :param verbose:
    :return: (n, m)
    """

    dim = len(mu)
    if M.shape[1] != dim:
        raise ValueError("The covariance matrix must be square.")

    if len(initial) != dim:
        raise ValueError("Wrong length for initial value vector.")

    # verify that M is positive definite, it will raise an error if M is not SPD
    R = np.linalg.cholesky(M)

    # we change variable to the canonical frame, and transform back after sampling
    # X ~ N(mu, M), then R^-1(X-mu) ~ N(0, I)
    init_trans = scipy.linalg.solve(R, initial - mu)  # the new initial value

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

        hmc = HmcSampler(dim, init_trans, f_trans, g_trans, verbose=verbose)
    else:
        hmc = HmcSampler(dim, init_trans, f, g, verbose=verbose)

    samples = np.zeros((n, dim))
    for i in range(burn_in):
        if verbose:
            print("="*30 + " (burn in) sample {} ".format(i) + "="*30)
        hmc.sampleNext()
    for i in range(n):
        if verbose:
            print("=" * 30 + " sample {} ".format(i) + "=" * 30)
        samples[i] = hmc.sampleNext()

    # transform back
    return samples @ R.T + mu
