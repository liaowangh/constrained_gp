import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import scipy.stats

from HmcSampler import tmg
from MHSampler import mh
from rtmg import py_rtmg
from RSM import rsm


class ConstrainedGP:
    def __init__(self, m, constraints=None, interval=None, kernel="SE", basis="spikes"):
        """

        :param m: number of nodes, if interval=[a,b], then t_j=(j-1)*(b-a)/(m-1), j=1,2,...,m
        :param constraints: dictionary, its keys are
                            increasing : boolean, true or false
                            bounded    : empty or (2,) indicates the bound
                            convex     : boolean, true or false
        :param interval: (2,)
        :param kernel: the kernel (covariance) function, e.g. "SE" stands for squared exponential
        :param basis: types of basis function, e.g. "spikes"
        """
        if interval is None:
            interval = [0, 1]
        if constraints is None:
            constraints = {'increasing': False, 'bounded': [], 'convex': False}
        self.interval = interval
        self.m = m
        self.constraints = constraints
        self.basis = basis
        self.kernel = kernel
        self.samples = None
        self.mean = None
        self.var = None

    def k(self, x, y):
        """
        The covariance function
        :param x:
        :param y:
        :return:
        """
        res = 0
        if self.kernel == "SE":
            alpha = 0.2
            sigma = 1
            res = sigma ** 2 * math.exp(-(x - y) ** 2 / (2 * alpha ** 2))
        return res

    def inequality_constraints(self):
        """
        Return the l, Lambda, u in the inequality conditions l <= Lambda* xi  <= u

        :return:
            l      : (q,1)
            Lambda : (q,m)
            u      : (q,1)
            There are q linear inequalities of the form
            l_k <= sum Lambda_kj*cj <= uk
        """
        m = self.m
        constraints = self.constraints
        increasing = constraint['increasing']
        convex = constraint['convex']
        bounded = len(constraint['bounded']) > 0
        l = None
        Lambda = None
        u = None
        if increasing and not bounded and not convex:
            # increasing
            # c1 <= c2 <= ... <= cm
            l = np.full(m, -np.inf)

            u = np.zeros(m)
            u[m - 1] = np.inf

            Lambda = np.identity(m) + np.diag(np.full(m - 1, -1), 1)
        elif not increasing and bounded and not convex:
            # bounded
            # l <= ci <= u
            l = np.full(m, constraints['bounded'][0])

            u = np.full(m, constraints['bounded'][1])

            Lambda = np.identity(m)
        elif not increasing and bounded and convex:
            # convex
            # cj - c_(j-1) >= c_(j-1) - c_(j-2)
            # c_(j-2) - 2*c_(j-1) + cj >= 0
            l = np.zeros(m)
            l[0] = -np.inf
            l[m - 1] = -np.inf

            u = np.full(m, np.inf)

            Lambda = np.identity(m) + np.diag(np.full(m - 1, 1), 1) + np.diag(np.full(m - 1, 1), -1)
            Lambda[0, 1] = 0
            Lambda[m - 1, m - 2] = 0
            for i in range(1, m - 1):
                Lambda[i, i] = -2
        elif increasing and bounded and not convex:
            # increasing and bounded
            # l <= c1 <= c2 <= ... <= cm <= u
            # l-u <= ci-c_(i+1) <= 0
            l = np.full(m + 1, -np.inf)
            l[0] = constraints['bounded'][0]

            u = np.zeros(m + 1)
            u[0] = np.inf
            u[m] = constraints['bounded'][1]

            Lambda = -1 * np.identity(m) + np.diag(np.ones(m - 1), -1)
            Lambda[0, 0] = 1
            Lambda = np.vstack((Lambda, np.zeros(m)))
            Lambda[m, m - 1] = 1
        elif increasing and not bounded and convex:
            # increasing and convex
            # cj - c_(j-1) >= c_(j-1) - c_(j-2)
            # c2 >= c1
            l = np.zeros(m)
            l[m - 1] = -np.inf

            u = np.full(m, np.inf)

            Lambda = np.identity(m) + np.diag(np.full(m - 1, 1), 1) + np.diag(np.full(m - 1, 1), -1)
            Lambda[m - 1, m - 2] = 0
            Lambda[0, 0] = -1
            for i in range(1, m - 1):
                Lambda[i, i] = -2

        elif not increasing and bounded and convex:
            # bounded and convex
            # 2(u-l) >= c_(j-2) - 2*c_(j-1) + cj >= 0
            Lambda1 = np.identity(m)
            l1 = np.full(m, constraints['bounded'][0])
            u1 = np.full(m, constraints['bounded'][1])

            Lambda2 = np.zeros((m - 2, m))
            for i in range(m - 2):
                Lambda2[i, i:i + 3] = np.array([1, -2, 1])
            l2 = np.zeros(m - 2)

            u2 = np.full(m - 2, np.inf)

            Lambda = np.vstack((Lambda1, Lambda2))
            l = np.concatenate((l1, l2), axis=None)
            u = np.concatenate((u1, u2), axis=None)

        elif increasing and bounded and convex:
            # increasing, bounded and convex
            # c2 - c1 >= 0
            # c_(i-2) - 2c_(i-1) + c_i >= 0
            # l <= c1
            #      cm <= u
            Lambda = np.zeros((m + 1, m))
            for i in range(m - 2):
                Lambda[i, i:i + 3] = np.array([1, -2, 1])
            Lambda[m - 2, 0] = -1
            Lambda[m - 2, 1] = 1
            Lambda[m - 1, 0] = 1
            Lambda[m, m - 1] = 1

            l = np.zeros(m + 1)
            l[m - 1] = constraints['bounded'][0]
            l[m] = -np.inf

            u = np.full(m+1, np.inf)
            u[m] = constraints['bounded'][1]
        else:
            pass
        return l, Lambda, u

    def basis_fun(self, x, j):
        """
        Return the value of basis function \phi_j(x)

        :param x: double, need to be in the interval
        :param j: integer, index of hat functions, 1 <= j <= m
        :return: \phi_j(x)
        """
        if (x < self.interval[0]).any() or (x > self.interval[1]).any():
            raise ValueError('x must be in the range of input interval')
        res = 0
        if self.basis == 'spikes':
            dm = (self.interval[1] - self.interval[0]) / (self.m - 1)  # delta m
            tj = (j - 1) * dm
            res = 1 - np.abs((x - tj) / dm)
            res[res < 0] = 0
        else:
            pass
        return res

    def interpolation_constraints(self, x):
        """
        Return coefficient matrix in the Interpolation conditions

        :param x: (n,), array_like, the design of experiment [x1,x2,...,xn]
        :return: (n,m), with entry phi_j(xi)
        """
        return np.array([self.basis_fun(x, j) for j in range(1, self.m + 1)]).transpose()

    def covariance(self):
        """

        :return: (m,m), the covariance matrix (k(t_i,t_j)),
                        t_j=j*(b-a)/(m-1), j=0,1,...,m-1
        """

        def t(j):
            return j * (self.interval[1] - self.interval[0]) / (self.m - 1)

        Gamma = [[self.k(t(i), t(j)) for j in range(self.m)] for i in range(self.m)]
        return np.array(Gamma)

    def mode(self, x, y, alpha=None):
        """
        :param x:     (n,), array_like, the design of experiment [x1,x2,...,xn]
        :param y:     (n,), array_like, the true value at x
        :param alpha: Gamma = Gamma + alpha * np.diag(np.diag(Gamma))

        :return :
        mu:    (m,) the posterior mean with interpolation condition only, Gamma*Phi^T*[Phi*Gamma*Phi^T]^-1*y
        Sigma: (m,m), the covariance matrix of the posterior with interpolation condition only,
                      Sigma = Gamma-Gamma*Phi^T*[Phi*Gamma*Phi^T]^-1*Phi*Gamma
        :mode: (m,) the posterior mode which is given by the maximum of the PDF of the posterior
        """
        l, Lambda, u = self.inequality_constraints()
        Phi = self.interpolation_constraints(x)
        Gamma = self.covariance()
        if alpha is not None:
            Gamma = Gamma + alpha * np.eye(self.m)
        mu = Gamma @ Phi.T @ scipy.linalg.solve(Phi @ Gamma @ Phi.T, y, assume_a='sym')
        Sigma = Gamma - Gamma @ Phi.T @ scipy.linalg.solve(Phi @ Gamma @ Phi.T, Phi, assume_a='sym') @ Gamma
        if alpha is not None:
            Sigma = Sigma + alpha * np.eye(self.m)

        if Lambda is None:
            # no inequality constraints, so mode is the posterior mean
            # mode = mu
            xi = cp.Variable(self.m)
            # obj = cp.Minimize(cp.quad_form(xi, scipy.linalg.inv(Gamma)))
            obj = cp.Minimize(cp.matrix_frac(xi, Gamma))
            constraints = [Phi @ xi == y]
            prob = cp.Problem(obj, constraints)
            prob.solve()
            mode = xi.value
        else:
            # CVXPY
            xi = cp.Variable(self.m)
            # obj = cp.Minimize(cp.quad_form(xi, scipy.linalg.inv(Gamma)))
            obj = cp.Minimize(cp.matrix_frac(xi, Gamma))
            constraints = [Phi @ xi == y]
            if not np.all(l == -np.inf):
                constraints.append(Lambda[l != -np.inf] * xi >= l[l != -np.inf])
            if not np.all(u == np.inf):
                constraints.append(Lambda[u != np.inf] * xi <= u[u != np.inf])

            prob = cp.Problem(obj, constraints)
            # print("Problem is DCP: ", prob.is_dcp())
            prob.solve()
            print("status:", prob.status)
            if prob.status != "optimal":
                raise ValueError('cannot compute the mode')
            mode = xi.value

            # def poster(x):
            #     y = np.linalg.solve(Gamma, x)
            #     return np.dot(y, x)
            #
            # def poster_gradient(x):
            #     return 2 * scipy.linalg.solve(Gamma, x)

            # def poster_hess(x):
            #     return 2 * np.linalg.inv(Gamma)
            # method='SLSQP'
            # cons = []
            # eq_cons = {'type': 'eq',
            #            'fun': lambda x: Phi @ x - y,
            #            'jac': lambda x: Phi}
            # cons.append(eq_cons)
            # if not np.all(l == -np.inf):
            #     ineq_cons1 = {'type': 'ineq',
            #                   'fun': lambda x: Lambda[l != -np.inf] @ x - l[l != -np.inf],
            #                   'jac': lambda x: Lambda[l != -np.inf]}
            #     cons.append(ineq_cons1)
            # if not np.all(u == np.inf):
            #     ineq_cons2 = {'type': 'ineq',
            #                   'fun': lambda x: u[u != np.inf] - Lambda[u != np.inf] @ x,
            #                   'jac': lambda x: -Lambda[u != np.inf]}
            #     cons.append(ineq_cons2)
            # res = minimize(poster, mu, method='SLSQP', jac=poster_gradient,
            #                constraints=cons, options={'ftol': 1e-9, 'disp': True})
            # mode = res.x
        return mu, Sigma, mode

    def interpolate(self, xi, x):
        """
        :param x:
        :param xi: (m,) the coefficients of basis function
        :return: the value of interpolation function at x,
                 function f = sum xi_j * phi_j(x)
        """
        res = 0
        for i in range(self.m):
            res += xi[i] * self.basis_fun(x, i + 1)
        return res

    def fit_gp(self, x, y, n=500, burn_in=100, alpha=0.0001, verbose=False, method='HMC'):
        """

        :param x: (n,), array_like, the design of experiment [x1,x2,...,xn]
        :param y: (n,)
        :param n: Number of samples.
        :param burn_in: The number of burn-in iterations in MCMC. The Markov chain is sampled n + burn_in
                        times, and the last n samples are returned.
        :param alpha:   coefficient of diagonal shift
        :param verbose:
        :param method:  sampling method. 'HMC' : Hamiltonian Monte Carlo. 'RSM' : reject sampling method.
                        'ET' : minimax exponential tilting. 'Gibbs' : Gibbs sampling. 'MH' : Metropolis-Hastings
        :return: (n,m) each row is a sample
        """
        mu, Sigma, initial = self.mode(x, y, alpha=alpha)

        l, Lambda, u = self.inequality_constraints()
        if Lambda is not None:
            # Lambda * xi - l >= 0
            # -Lambda * xi + u >= 0

            f = np.vstack((np.eye(len(l)), -np.eye(len(u))))
            g = np.hstack((-l, u))

            R = Lambda @ Sigma @ Lambda.T
            if alpha is not None:
                R = R + alpha * np.eye(len(R))

            eta = Lambda@initial  # new initial value, constraints: eta >= l, eta <= u,
            # however the constraints may not be satisfied due to numerical issue
            eta[eta < l] = l[eta < l] + 1e-8
            eta[eta > u] = u[eta > u] - 1e-8

            if method == 'HMC':
                samples = py_rtmg(n, Lambda@mu, R, eta, f, g, burn_in=burn_in, verbose=verbose)
            elif method == 'RSM':
                samples = rsm(n, Lambda@mu, R, f, g, verbose=verbose)
            elif method == 'MH':
                samples = mh(n, Lambda@mu, R, eta, f, g, 1, burn_in=burn_in)
            else:
                raise ValueError(method + " is not supported.")
            samples = scipy.linalg.solve(Lambda.T @ Lambda, Lambda.T) @ samples.T
            samples = samples.T
        else:
            # we have analytic formulation of the posterior distribution
            if method == 'HMC':
                # samples = tmg(n, Sigma, mu, initial, None, None, burn_in=burn_in)
                samples = py_rtmg(n, mu, Sigma, initial, None, None, burn_in=burn_in)
            elif method == 'RSM':
                samples = rsm(n, mu, Sigma, None, None, verbose=verbose)
            elif method == 'MH':
                samples = mh(n, mu, Sigma, initial, None, None, 1, burn_in=burn_in)
            else:
                raise ValueError(method + " is not supported.")

        # print(samples)
        self.samples = samples
        return samples

    def mean_var(self, xtest):
        """
        the conditional mean at xtest
        :param xtest: (k,) k test points
        """
        assert self.samples is not None, "Has not fit yet."
        # conditional_mean = np.mean(self.samples, axis = 0)
        # return self.interpolate(conditional_mean, xtest)
        coeff_phi = [self.basis_fun(xtest, j + 1) for j in range(self.m)]  # (m,k)
        coeff_phi = np.array(coeff_phi).T  # (k, m)
        y_sample = coeff_phi @ self.samples.T  # (k, n)
        self.mean = np.mean(y_sample, axis=1)
        self.var = np.var(y_sample, axis=1)
        return self.mean

    # def var(self, xtest):
    #     """
    #     the conditional variance at xtest
    #     """
    #     assert self.samples is not None, "Has not fit yet."
    #     coeff_phi = [self.basis_fun(xtest, j + 1) for j in range(self.m)]  # (m,k)
    #     coeff_phi = np.array(coeff_phi).T  # (k, m)
    #     y_sample = coeff_phi@self.samples.T   # (k, n)
    #     self.var = np.var(y_sample, axis = 1)
    #     return self.var

    def confidence_interval(self, confidence=0.9):
        assert self.mean is not None and self.var is not None, "Need to have mean and var first."
        n = self.samples.shape[0]
        # se = np.sqrt(self.var)
        h = np.sqrt(self.var) * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return self.mean - h, self.mean + h

    # def mean_var(self, xtest):
    #     """
    #     more efficient joint calculation
    #     xtest is (n,d)
    #     """


def condition_num(const, x, alpha=None):
    print("{} {} {} {} {} {} {} {}".format("  m", "rank_Gamma", "cond_Gamma", "PSD_sigma", "rank_sigma", "cond_sigma",
                                           "PSD_Rinv", "cond_Rinv"))
    for m in range(10, 105, 5):
        Gp = ConstrainedGP(m, constraints=const)
        Gamma = Gp.covariance()
        if alpha is not None:
            Gamma = Gamma + alpha * np.eye(len(Gamma))
        Phi = Gp.interpolation_constraints(x)
        l, Lambda, u = Gp.inequality_constraints()
        Sigma = Gamma - Gamma @ Phi.T @ scipy.linalg.solve(Phi@Gamma@Phi.T, Phi, assume_a='sym') @ Gamma
        if alpha is not None:
            Sigma = Sigma + alpha * np.eye(len(Sigma))

        R = Lambda@Sigma@Lambda.T

        if alpha is not None:
            R = R + alpha * np.eye(len(R))
        Rinv = scipy.linalg.inv(R)

        cond_Gamma = np.linalg.cond(Gamma)
        rank_Gamma = np.linalg.matrix_rank(Gamma)
        cond_Sigma = np.linalg.cond(Sigma)
        rank_Sigma = np.linalg.matrix_rank(Sigma)
        PSD_Sigma = np.all(np.linalg.eigvals(Sigma) > 0)
        cond_Rinv = np.linalg.cond(Rinv)
        PSD_Rinv = np.all(np.linalg.eigvals(Rinv) > 0)
        print("%3d %5d      %5.4E %5d       %5d    %5.4E %4d     %5.4E"
              % (m, rank_Gamma, cond_Gamma, PSD_Sigma, rank_Sigma, cond_Sigma, PSD_Rinv, cond_Rinv))


def plot_mode(m):
    const = [{'increasing': False, 'bounded': [], 'convex': False},
             {'increasing': False, 'bounded': [0, 1], 'convex': False},
             {'increasing': True, 'bounded': [], 'convex': False},
             {'increasing': True, 'bounded': [0, 1], 'convex': False}]
    const_type = ["Unconstrained", "Boundedness", "Monotonicity", "Boundedness and monotonicity"]

    rv = norm()

    def f(x):
        return rv.cdf((x - 0.5) / 0.2)

    x_train = np.array([0.25, 0.5, 0.75])
    y_train = f(x_train)

    t = np.arange(0, 1 + 0.01, 0.01)
    y_true = f(t)

    fig, axs = plt.subplots(2, 2)
    for i in range(4):
        Gp = ConstrainedGP(m, constraints=const[i])
        a, b, mode = Gp.mode(x_train, y_train, alpha=0.0001)

        def f_mode(x):
            return Gp.interpolate(mode, x)

        y_mode = f_mode(t)

        axs[i // 2, i % 2].plot(t, y_true, 'r', label="true function")
        axs[i // 2, i % 2].plot(t, y_mode, 'b', label="mode")
        axs[i // 2, i % 2].plot(x_train, y_train, 'ko', label="training points")
        axs[i // 2, i % 2].legend(loc='upper left')
        axs[i // 2, i % 2].set_title(const_type[i])

    plt.show()


def title(constraint):
    increasing = constraint['increasing']
    convex = constraint['convex']
    bounded = len(constraint['bounded']) > 0

    if increasing and not convex and not bounded:
        title = 'Monotonicity'
    elif increasing and not convex and bounded:
        title = 'Monotonicity and boundedness'
    elif increasing and convex and not bounded:
        title = 'Monotonicity and convexity'
    elif increasing and convex and bounded:
        title = 'Monotonicity, boundedness and convexity'
    elif not increasing and not convex and bounded:
        title = 'Boundedness'
    elif not increasing and convex and not bounded:
        title = 'Convexity'
    elif not increasing and convex and bounded:
        title = 'Convexity and boundedness'
    else:
        title = 'Unconstrained'
    return title


if __name__ == "__main__":
    # plot_mode(100)
    constraint = {'increasing': True, 'bounded': [], 'convex': False}
    sampling_method = 'RSM'
    # condition_num(constraint, np.array([0.25, 0.5, 0.75]), 0.0000001)

    Gp = ConstrainedGP(m=30, constraints=constraint)
    rv = norm()

    def f(x):
        return rv.cdf((x - 0.5) / 0.2)
    x_train = np.array([0.25, 0.5, 0.75])
    # def f(x):
    #     return x*x
    # x_train = np.array([0.1, 0.5, 0.9])
    y_train = f(x_train)
    samples = Gp.fit_gp(x_train, y_train, n=500, burn_in=100, alpha=0.0000001,
                        verbose=True, method=sampling_method)

    t = np.arange(0, 1 + 0.01, 0.01)
    y_true = f(t)

    Gp.mean_var(t)

    y_pred = Gp.mean
    ci_l, ci_u = Gp.confidence_interval()

    fig, axis = plt.subplots()
    axis.plot(t, y_true, 'r', label="true function")
    axis.plot(t, y_pred, 'b', label="sample")
    axis.fill_between(t, ci_l, ci_u, color='lightgrey')
    axis.plot(x_train, y_train, 'ko', label="training points")
    axis.set_xlim(0., 1.)
    axis.set_ylim(0., 1.)
    axis.set_xlabel('x')
    axis.set_ylabel('y(x)')
    axis.legend(loc='upper left')
    axis.set_title(title(constraint)+',method: ' + sampling_method)

    plt.show()
