import cvxpy as cp
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm
import scipy.stats

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from GibbsSampler import gibbs
from HmcSampler import tmg
from MHSampler import mh
from RSM import rsm
from rtmg import py_rtmg
from utility import title


class ConstrainedGP:
    def __init__(self, m, constraints=None, interval=None, alpha=0.0000001, basis="spikes"):
        """

        :param m: scale or (2,), number of nodes in each dimension,
                  if interval=[a,b], then t_j=(j-1)*(b-a)/(m-1), j=1,2,...,m
        :param constraints: dictionary, its keys are
                            increasing : boolean, true or false
                            bounded    : empty or (2,) indicates the bound
                            convex     : boolean, true or false
        :param interval: (2,) or a list contains two intervals
        :param alpha: for a matrix A with large condition number,
                      let A = A + alpha * I to make A more stable in numerical computational
        :param basis: types of basis function, e.g. "spikes"
        """
        if constraints is None:
            constraints = {'increasing': False, 'bounded': [], 'convex': False}
        # m = np.array(m)
        # dim = m
        try:
            dim = len(m)
        except:
            dim = 1
        assert dim == 1 or dim == 2, "Only support 1D and 2D case"

        if interval is None:
            if dim == 1:
                interval = [0, 1]
            else:
                interval = [[0, 1], [0, 1]]

        self.dim = dim
        self.interval = interval
        self.m = m
        self.constraints = constraints
        self.alpha = alpha
        self.basis = basis
        self.samples = None
        self.mean = None
        self.var = None

    def k(self, x, y):
        """
        The covariance function. SE covariance function is used.
        :param x: scale or (2,)
        :param y: scale or (2,)
        :return:
        """
        if self.dim == 1:
            alpha = 0.2
            sigma = 1
            res = sigma ** 2 * math.exp(-(x - y) ** 2 / (2 * alpha ** 2))
        else:
            alpha1 = 0.2
            alpha2 = 0.2
            sigma = 1
            res = sigma ** 2 * math.exp(-(x[0]-y[0])**2/(2*alpha1**2) -
                                         (x[1]-y[1])**2/(2*alpha2**2))
        return res

    def inequality_constraints(self):
        """
        Return the l, Lambda, u in the inequality conditions l <= Lambda* xi  <= u

        :return:
            l      : (q,1)
            Lambda : (q,m) or (q, m1*m2)
            u      : (q,1)
            There are q linear inequalities of the form
            l_k <= sum Lambda_kj*cj <= uk
        """
        m = self.m
        constraints = self.constraints
        increasing = constraints['increasing']
        convex = constraints['convex']
        bounded = len(constraints['bounded']) > 0
        l = None
        Lambda = None
        u = None
        if increasing and not bounded and not convex:
            if self.dim == 1:
                # increasing
                # c1 <= c2 <= ... <= cm
                l = np.full(m, -np.inf)

                u = np.zeros(m)
                u[m - 1] = np.inf

                Lambda = np.identity(m) + np.diag(np.full(m - 1, -1), 1)
            else:
                m1 = m[0]
                m2 = m[1]
                total_con = m2*(m1-1) + m1*(m2-1)
                # total_con = 2*(m2-1)*(m1-1)
                l = np.full(total_con, -np.inf)
                u = np.zeros(total_con)

                Lambda = []
                for i in range(m1):
                    for j in range(m2-1):
                        tmp = np.zeros(m1*m2)
                        tmp[i*m2+j] = 1
                        tmp[i*m2+j+1] = -1
                        Lambda.append(tmp)
                for j in range(m2):
                    for i in range(m1-1):
                        tmp = np.zeros(m1*m2)
                        tmp[i*m2+j] = 1
                        tmp[(i+1)*m2+j] = -1
                        Lambda.append(tmp)
                Lambda = np.array(Lambda)

        elif not increasing and bounded and not convex:
            # bounded
            # l <= ci <= u
            if self.dim == 2:
                m = m[0]*m[1]
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

            u = np.full(m + 1, np.inf)
            u[m] = constraints['bounded'][1]
        else:
            pass
        return l, Lambda, u

    def basis_fun(self, x, j, which_dim=0):
        """
        Return the value of basis function \phi_j(x)

        :param x:         scale, need to be in the interval
        :param j:         integer, index of hat functions, 1 <= j <= m
        :param which_dim: 0 or 1, specify which dimension in 2D case.
                          0 for the first dimension, 1 for the second dimension

        :return: \phi_j(x)
        """
        if self.dim == 1:
            a = self.interval[0]  # lower bound
            b = self.interval[1]  # upper bound
            m = self.m
        else:
            a = self.interval[which_dim][0]
            b = self.interval[which_dim][1]
            m = self.m[which_dim]

        assert np.all(np.logical_and(a <= x, x <= b)), 'x must be in the range of input interval'
        dm = (b - a) / (m - 1)
        tj = a + (j - 1) * dm
        res = 1 - np.abs((x - tj) / dm)
        res = np.array(res)
        res[res < 0] = 0
        return res

    def interpolation_constraints(self, x):
        """
        Return coefficient matrix in the Interpolation conditions

        :param x: (n,) or (n,2), the design of experiment [x1,x2,...,xn]
        :return: 1D: (n,m), with entry phi_j(xi)
                 2D: (n, m1*m2)
        """
        if self.dim == 1:
            return np.array([self.basis_fun(x, j) for j in range(1, self.m + 1)]).T
        else:
            n = len(x)
            phi_1 = np.array([self.basis_fun(x[:, 0], j) for j in range(1, self.m[0] + 1)]).T
            phi_2 = np.array([self.basis_fun(x[:, 1], j) for j in range(1, self.m[1] + 1)]).T
            res = [np.kron(phi_1[i], phi_2[i]) for i in range(n)]
            res = np.array(res)
            # res = np.zeros((n, np.prod(self.m)))
            # for i in range(n):
            #     phi_1 = np.array([self.basis_fun(x[i][0], j, 0) for j in range(1, self.m[0] + 1)])
            #     phi_2 = np.array([self.basis_fun(x[i][1], j, 1) for j in range(1, self.m[1] + 1)])
            #     res[i] = np.kron(phi_1, phi_2)
            return res

    def covariance(self):
        """
        :return: (m,m), the covariance matrix (k(t_i,t_j)),
                        t_j = a + j*(b-a)/(m-1), j=0,1,...,m-1. or
                 (m1*m2, m1*m2),
        """
        if self.dim == 1:
            def t(j):
                return self.interval[0] + j * (self.interval[1] - self.interval[0]) / (self.m - 1)

            Gamma = [[self.k(t(i), t(j)) for j in range(self.m)] for i in range(self.m)]
        else:
            m1 = self.m[0]
            m2 = self.m[1]
            a1 = self.interval[0][0]
            a2 = self.interval[1][0]
            dm1 = (self.interval[0][1] - self.interval[0][0]) / (m1 - 1)
            dm2 = (self.interval[1][1] - self.interval[1][0]) / (m2 - 1)

            def t(k):
                i = k / m2
                j = k % m2
                return [a1 + i*dm1, a2 + j*dm2]
            Gamma = [[self.k(t(i), t(j)) for j in range(m1*m2)] for i in range(m1*m2)]
        return np.array(Gamma)

    def mode(self, x, y):
        """
        :param x:     (n,) or (n,2), the design of experiment [x1,x2,...,xn]
        :param y:     (n,), array_like, the true value at x

        :return : (in 2D, m = m1*m2)
        mu:    (m,) the posterior mean with interpolation condition only, Gamma*Phi^T*[Phi*Gamma*Phi^T]^-1*y
        Sigma: (m,m), the covariance matrix of the posterior with interpolation condition only,
                      Sigma = Gamma-Gamma*Phi^T*[Phi*Gamma*Phi^T]^-1*Phi*Gamma
        mode:  (m,) the posterior mode which is given by the maximum of the PDF of the posterior
        """
        l, Lambda, u = self.inequality_constraints()
        Phi = self.interpolation_constraints(x)
        Gamma = self.covariance()
        Gamma = Gamma + self.alpha * np.eye(len(Gamma))
        mu = Gamma @ Phi.T @ np.linalg.solve(Phi @ Gamma @ Phi.T, y)
        Sigma = Gamma - Gamma @ Phi.T @ np.linalg.solve(Phi @ Gamma @ Phi.T, Phi) @ Gamma
        Sigma = Sigma + self.alpha * np.eye(len(Sigma))

        if Lambda is None:
            # no inequality constraints, so mode is the posterior mean
            mode = mu
        else:
            xi = cp.Variable(np.prod(self.m))
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

        return mu, Sigma, mode

    def interpolate(self, xi, x):
        """
        :param x: scale or (2,)
        :param xi: (m,) or (m1*m2,), the coefficients of basis function

        :return: the value of interpolation function at x,
                 function f = sum xi_j * phi_j(x)  (1D) or
                 f = sum xi_kl * phi^1_k(x[1]) * phi^2_l(x[2])
        """
        if self.dim == 1:
            phi = [self.basis_fun(x, i + 1) for i in range(self.m)]
            return np.array(phi).dot(xi)
        else:
            phi_1 = np.array([self.basis_fun(x[0], j, 0) for j in range(self.m[0])])
            phi_2 = np.array([self.basis_fun(x[1], j, 1) for j in range(self.m[1])])
            return np.kron(phi_1, phi_2).dot(xi)

    def fit_gp(self, x, y, n=500, burn_in=100, method='HMC'):
        """

        :param x: (n,) or (n, 2), the design of experiment [x1,x2,...,xn]
        :param y: (n,)
        :param n: Number of samples.
        :param burn_in: The number of burn-in iterations in MCMC. The Markov chain is sampled n + burn_in
                        times, and the last n samples are returned.
        :param method:  sampling method. 'HMC' : Hamiltonian Monte Carlo. 'RSM' : reject sampling method.
                        'Gibbs' : Gibbs sampling. 'MH' : Metropolis-Hastings
        :return: (n,m) each row is a sample
        """
        mu, Sigma, initial = self.mode(x, y)

        l, Lambda, u = self.inequality_constraints()
        if Lambda is not None:
            # Lambda * xi - l >= 0
            # -Lambda * xi + u >= 0

            f = np.vstack((np.eye(len(l)), -np.eye(len(u))))
            g = np.hstack((-l, u))

            R = Lambda @ Sigma @ Lambda.T
            R = R + self.alpha * np.eye(len(R))

            eta = Lambda @ initial  # new initial value, constraints: eta >= l, eta <= u,
            # however the constraints may not be satisfied due to numerical issue
            eta[eta < l] = l[eta < l] + 1e-8
            eta[eta > u] = u[eta > u] - 1e-8

            if method == 'HMC':
                samples = tmg(n, Lambda @ mu, R, eta, f, g, burn_in=burn_in)
            elif method == 'RHMC':
                samples = py_rtmg(n, Lambda @ mu, R, eta, f, g, burn_in=burn_in)
            elif method == 'RSM':
                samples = rsm(n, Lambda @ mu, R, f, g)
            elif method == 'MH':
                samples = mh(n, Lambda @ mu, R, eta, f, g, 0.1, burn_in=burn_in)
            elif method == 'Gibbs':
                samples = gibbs(n, Lambda @ mu, R, eta, f, g, burn_in=burn_in)
            else:
                raise ValueError("Not supported method.")
            samples = np.linalg.solve(Lambda.T @ Lambda, Lambda.T) @ samples.T
            samples = samples.T
        else:
            # we have analytic formulation of the posterior distribution
            if method == 'HMC':
                samples = tmg(n, mu, Sigma, initial, None, None, burn_in=burn_in)
            elif method == "RHMC":
                samples = py_rtmg(n, mu, Sigma, initial, None, None, burn_in=burn_in)
            elif method == 'RSM':
                samples = rsm(n, mu, Sigma, None, None)
            elif method == 'MH':
                samples = mh(n, mu, Sigma, initial, None, None, 0.1, burn_in=burn_in)
            elif method == 'Gibbs':
                samples = gibbs(n, mu, Sigma, initial, None, None, burn_in=burn_in)
            else:
                raise ValueError("Not supported method.")
        # print(samples)
        self.samples = samples
        return samples

    def mean_var(self, xtest):
        """
        the conditional mean at xtest
        :param xtest: (k,) or (k, 2) k test points
        """
        assert self.samples is not None, "Has not fit yet."
        # conditional_mean = np.mean(self.samples, axis = 0)
        # return self.interpolate(conditional_mean, xtest)
        if self.dim == 1:
            coeff_phi = [self.basis_fun(xtest, j + 1) for j in range(self.m)]  # (m,k)
            coeff_phi = np.array(coeff_phi).T  # (k, m)
        else:
            n = len(xtest)
            phi_1 = np.array([self.basis_fun(xtest[:, 0], j) for j in range(1, self.m[0] + 1)]).T
            phi_2 = np.array([self.basis_fun(xtest[:, 1], j) for j in range(1, self.m[1] + 1)]).T
            coeff_phi = [np.kron(phi_1[i], phi_2[i]) for i in range(n)]
            coeff_phi = np.array(coeff_phi)
        y_sample = coeff_phi @ self.samples.T  # (k, n)
        self.mean = np.mean(y_sample, axis=1)
        self.var = np.var(y_sample, axis=1)
        return self.mean

    def confidence_interval(self, confidence=0.9):
        assert self.mean is not None and self.var is not None, "Need to have mean and var first."
        n = self.samples.shape[0]
        h = np.sqrt(self.var) * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return self.mean - h, self.mean + h


def plot_fig2(m, method, n, burn_in):
    const = [{'increasing': False, 'bounded': [], 'convex': False},
             {'increasing': False, 'bounded': [0, 1], 'convex': False},
             {'increasing': True, 'bounded': [], 'convex': False},
             {'increasing': True, 'bounded': [0, 1], 'convex': False}]

    rv = norm()

    def f(x):
        return rv.cdf((x - 0.5) / 0.2)

    x_train = np.array([0.25, 0.5, 0.75])
    y_train = f(x_train)
    t = np.arange(0, 1 + 0.01, 0.01)
    y_true = f(t)

    fig, axs = plt.subplots(2, 2)
    for i in range(3):
        Gp = ConstrainedGP(m, constraints=const[i])
        Gp.fit_gp(x_train, y_train, n=n, burn_in=burn_in, method=method)
        Gp.mean_var(t)

        y_pred = Gp.mean
        ci_l, ci_u = Gp.confidence_interval()

        axs[i // 2, i % 2].plot(t, y_true, 'r', label="true function")
        axs[i // 2, i % 2].plot(t, y_pred, 'b', label="sample")
        axs[i // 2, i % 2].fill_between(t, ci_l, ci_u, color='lightgrey')
        axs[i // 2, i % 2].plot(x_train, y_train, 'ko', label="training points")
        axs[i // 2, i % 2].set_xlim(0., 1.)
        axs[i // 2, i % 2].set_ylim(0., 1.)
        axs[i // 2, i % 2].set_xlabel('x')
        axs[i // 2, i % 2].set_ylabel('y(x)')
        axs[i // 2, i % 2].legend(loc='upper left')
        axs[i // 2, i % 2].set_title(title(const[i]) + ', method: ' + method)

    plt.show()


def plot_fig(m, constraint, interval, method, n, burn_in):
    Gp = ConstrainedGP(m, constraints=constraint, interval=interval, alpha=0.0001)

    # def f(x1, x2):
    #     return -0.5*(np.sin(9*x1)-np.cos(9*x2))
    def f(x1, x2):
        return np.arctan(5*x1)+np.arctan(x2)

    x = np.arange(0, 1.1, 0.2)
    y = np.arange(0, 1.1, 0.2)
    x, y = np.meshgrid(x, y)
    x_train = np.array([x.flatten(), y.flatten()]).T
    z_train = f(x, y)

    Gp.fit_gp(x_train, z_train.flatten(), n=n, burn_in=burn_in, method=method)

    t = np.arange(0, 1, 0.01)
    t1, t2 = np.meshgrid(t, t)
    x_test = np.array([t1.flatten(), t2.flatten()]).T
    z_true = f(t1, t2)

    Gp.mean_var(x_test)

    z_pred = Gp.mean
    z_pred = z_pred.reshape((100, 100))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(t1, t2, z_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-0.1, np.pi - 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.scatter(x, y, z_train)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y(x1,x2)')

    plt.show()


def condition_num(const, alpha=0.0001):
    print("{} {} {} {} {} {} {} {}".format("  m", "rank_Gamma", "cond_Gamma", "PSD_sigma", "rank_sigma", "cond_sigma",
                                           "PSD_Rinv", "cond_Rinv"))
    x = np.arange(0, 1, 0.25)
    y = np.arange(0, 1, 0.25)
    x, y = np.meshgrid(x, y)
    x = np.array([x.flatten(), y.flatten()]).T

    for m in range(4, 13, 1):
        Gp = ConstrainedGP([m, m], constraints=const)
        Gamma = Gp.covariance()
        if alpha is not None:
            Gamma = Gamma + alpha * np.eye(len(Gamma))
        Phi = Gp.interpolation_constraints(x)
        l, Lambda, u = Gp.inequality_constraints()
        Sigma = Gamma - Gamma @ Phi.T @ np.linalg.solve(Phi @ Gamma @ Phi.T, Phi) @ Gamma
        if alpha is not None:
            Sigma = Sigma + alpha * np.eye(len(Sigma))

        R = Lambda @ Sigma @ Lambda.T

        if alpha is not None:
            R = R + alpha * np.eye(len(R))
        Rinv = np.linalg.inv(R)

        cond_Gamma = np.linalg.cond(Gamma)
        rank_Gamma = np.linalg.matrix_rank(Gamma)
        cond_Sigma = np.linalg.cond(Sigma)
        rank_Sigma = np.linalg.matrix_rank(Sigma)
        PSD_Sigma = np.all(np.linalg.eigvals(Sigma) > 0)
        cond_Rinv = np.linalg.cond(Rinv)
        PSD_Rinv = np.all(np.linalg.eigvals(Rinv) > 0)
        print("%3d %5d      %5.4E %5d       %5d    %5.4E %4d     %5.4E"
              % (m, rank_Gamma, cond_Gamma, PSD_Sigma, rank_Sigma, cond_Sigma, PSD_Rinv, cond_Rinv))


if __name__ == "__main__":
    constraint = {'increasing': True, 'bounded': [], 'convex': False}
    sampling_method = 'RSM'

    # plot_fig(30, constraint, sampling_method, 100, 100)
    plot_fig2(30, sampling_method, 500, 300)

    # m = [5, 5]
    # interval = [[0, 1], [0, 1]]
    # Gp = ConstrainedGP(m, constraint, interval)
    # plot_fig(m, constraint, interval, sampling_method, 50, 50)


