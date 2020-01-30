import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


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


def plot_fig(m, constraint, method, n, burn_in):
    Gp = ConstrainedGP(m, constraints=constraint)
    rv = norm()

    def f(x):
        return rv.cdf((x - 0.5) / 0.2)

    x_train = np.array([0.25, 0.5, 0.75])
    y_train = f(x_train)
    Gp.fit_gp(x_train, y_train, n=n, burn_in=burn_in, alpha=0.0000001,
              verbose=False, method=method)

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
    axis.set_title(title(constraint) + ',method: ' + method)

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
        Sigma = Gamma - Gamma @ Phi.T @ np.linalg.solve(Phi @ Gamma @ Phi.T, Phi, assume_a='sym') @ Gamma
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
    for i in range(4):
        Gp = ConstrainedGP(m, constraints=const[i])
        Gp.fit_gp(x_train, y_train, n=n, burn_in=burn_in, alpha=0.0000001,
                  verbose=False, method=method)
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