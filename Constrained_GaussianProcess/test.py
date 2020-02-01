import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from pyDOE import lhs
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from Constrained_GaussianProcess import ConstrainedGP
from Constrained_GaussianProcess import tmg


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


def plot_fig1D(m, method, n, burn_in):
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
    for i in range(4):
        Gp = ConstrainedGP(m, constraints=const[i])
        Gp.fit_gp(x_train, y_train, n=n, burn_in=burn_in, method=method)
        y_pred = Gp.mean(t)

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


def plot_fig2D(m, constraint, interval, method, n, burn_in):
    Gp = ConstrainedGP(m, constraints=constraint, interval=interval, alpha=0.0001)

    # def f(x1, x2):
    #     return -0.5*(np.sin(9*x1)-np.cos(9*x2))
    def f(x1, x2):
        return np.arctan(5*x1)+np.arctan(x2)

    # x = np.arange(0, 1.1, 0.2)
    # y = np.arange(0, 1.1, 0.2)
    # x, y = np.meshgrid(x, y)
    # x_train = np.array([x.flatten(), y.flatten()]).T
    # z_train = f(x, y)
    x_train = lhs(2, 15)
    z_train = f(x_train[:, 0], x_train[:, 1])

    Gp.fit_gp(x_train, z_train.flatten(), n=n, burn_in=burn_in, method=method)

    t = np.arange(0, 1, 0.01)
    t1, t2 = np.meshgrid(t, t)
    x_test = np.array([t1.flatten(), t2.flatten()]).T
    z_true = f(t1, t2)

    z_pred = Gp.mean(x_test)
    z_pred = z_pred.reshape((100, 100))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(t1, t2, z_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-0.1, 2.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.scatter(x_train[:, 0], x_train[:, 1], z_train)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y(x1,x2)')

    ax.set_title(title(constraint) + ', method: ' + method)

    plt.show()


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
    sampling_method = 'HMC'

    # plot_fig1D(30, sampling_method, 100, 100)

    m = [10, 10]
    interval = [[0, 1], [0, 1]]
    Gp = ConstrainedGP(m, constraint, interval)
    plot_fig2D(m, constraint, interval, sampling_method, 50, 50)

    """
    n = 150
    burn_in = 30

    # Define the covariance matrix and mean vector
    M = np.array([[0.5, -0.4], [-0.4, 0.5]])
    mu = np.array([0, 0])

    # Set initial point for the Markov chain
    initial = np.array([4, 1])

    # Define two linear constraints
    f = np.array([[1, 1], [1, 0]])
    g = np.array([0, 0])

    # Sample
    samples = tmg(n, mu, M, initial, f, g, burn_in=burn_in)
    print(samples)
    """
