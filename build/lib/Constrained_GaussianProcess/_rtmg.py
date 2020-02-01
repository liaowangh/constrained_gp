# Hamiltonian Monte Carlo by calling R function 'rtmg'

import numpy as np
import scipy.linalg
from rpy2 import robjects
from rpy2.robjects.packages import importr

if not robjects.packages.isinstalled('tmg'):
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('tmg')


def np2r(x):
    nr, nc = x.shape
    xvec = robjects.FloatVector(x.transpose().reshape(x.size))
    xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc)
    return xr


def py_rtmg(n, mu, Sigma, initial, f=None, g=None, burn_in=30):
    """
    This function generates samples from a Markov chain whose equilibrium distribution is a d-dimensional
    multivariate Gaussian truncated by linear inequalities. The probability log density is
    log p(X) = - 0.5 X^T M X + r^T X + const
    in terms of a precision matrix M and a vector r. The constraints are imposed as explained below.
    The Markov chain is built using the Hamiltonian Monte Carlo technique.

    The input M and mu are covariance matrix and mean, so we transform them into precision matrix and
    linear coefficient first.
    M = Sigma^-1
    r = M*mu

    :param n:       Number of samples.
    :param mu:      (m,) vector for the mean of multivariate Gaussian density
    :param Sigma:   (m,m) covariance matrix of the multivariate Gaussian density
    :param initial: (m,) vector with the initial value of the Markov chain. Must satisfy
                    the truncation inequalities strictly.
    :param f:       (q,m) matrix, where q is the number of linear constraints. The constraints require each component
                    of the m-dimensional vector fX+g to be non-negative
    :param g:       (q,) vector with the constant terms in the above linear constraints.
    :param burn_in: The number of burn-in iterations. The Markov chain is sampled n + burn_in
                    times, and the last n samples are returned.
    :return: (n, m)
    """
    tmg = importr('tmg')

    M = scipy.linalg.inv(Sigma)
    r = M@mu

    n = robjects.IntVector([n])
    M = np2r(M)
    r = robjects.FloatVector(r)

    initial = robjects.FloatVector(initial)
    burn_in = robjects.IntVector([burn_in])
    if f is None and g is None:
        res = np.array(tmg.rtmg(n, M, r, initial, burn_in=burn_in))
    else:
        # g man contains infinity, extract valid constraints
        valid = np.logical_and(g < np.inf, g > -np.inf)
        g = g[valid]
        f = f[valid]
        if not np.all(f@initial+g >= 0):
            raise ValueError("initial value does not satisfy the constraints.")
        f = np2r(f)
        g = robjects.FloatVector(g)
        res = np.array(tmg.rtmg(n, M, r, initial, f, g, burn_in=burn_in))

    return res
