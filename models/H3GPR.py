__author__ = 'morgan'
import numpy as np
from H3Kernels import kernel as kernel


class H3GPR(object):
    '''
    Gaussian process regression main class
    The implementation is referred to Rasmussen & Williams' GP book.
    '''

    def __init__(self, sigma=0.5, kernel='rbf', gamma=0.5,
                 coef0=1, degree=2):
        '''
        object constructor
        :param
            sigma: noise level
            kernel: string type of kernel, {'rbf','linear','polynomial'}
            gamma: e.g., in rbf kernel: exp(-gamma|xj-xi|^2)
            coef0: for linear kernel, e.g., coef0*XX^T
            degree: for polynomial, e.g., |xi-xj|^(degree)
        :return:
        '''
        self.sigma = sigma
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def fit(self, X, y, Xt):
        '''
        fit function of Gaussian process regression
        :param X: observations
        :param y: observation targets
        :param Xt: test samples
        :return: predictive mean, covariance on test samples, and marginal log-likelihood on training set
        '''
        n, d = X.shape
        K = kernel(X, metric=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree,
                   filter_params=True) + 1e-3
        K_xt = kernel(X, Xt, metric=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree,
                      filter_params=True)
        K_tt = kernel(Xt, metric=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree,
                      filter_params=True)
        L = np.linalg.cholesky(K + self.sigma * np.eye(n))
        K_inv = np.linalg.inv(K + self.sigma * np.eye(n))
        alpha = K_inv.dot(y.reshape(n, 1))
        m = K_xt.T.dot(alpha).ravel()
        S = K_tt - K_xt.T.dot(K_inv).dot(K_xt)
        log_likelihood = -.5 * \
            y.reshape(1, n).dot(alpha) - np.log(L.diagonal()
                                                ).sum() - .5 * n * np.log(2 * np.pi)
        return m, S, log_likelihood


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def obj_f(x):
        # 1d objective function
        return np.array([np.sin(x0)**2 + x0 * np.cos(x0) - .1 * x0**2 for x0 in x])

    X = np.linspace(-5, 5, 100).reshape(100, 1)
    y = obj_f(X)
    given_idx = np.random.choice(y.size, 10)
    Xtr = X[given_idx]
    ytr = y[given_idx]
    fig = plt.figure(figsize=(16,8))
    for id, g in enumerate(np.linspace(0.1, 10, 4)):
        gpr = H3GPR(sigma=1e-3, gamma=g)
        m, S, LL = gpr.fit(Xtr, ytr, X)
        plt.subplot(2, 2, id + 1)
        # randomly sample functions from predictive distribution
        # f = np.random.multivariate_normal(m, S, size=10)
        # plt.plot(X, f[0], 'r-')
        # plt.plot(X, f[1], 'm-')
        # plt.plot(X, f[2], 'g-')
        plt.plot(X, y, 'k-.', label='groundtruth f', lw=0.5)
        plt.plot(X, m, 'b-', label='predictive mean', lw=1)
        plt.plot(X[given_idx], y[given_idx], 'k+',
                 ms=10, mfc='none', mec='k', mew=1)
        plt.fill_between(X.ravel(), m - 2 * S.diagonal(), m + 2 * S.diagonal(),
                         facecolor='gray', alpha=.2, edgecolor='none')
        plt.gca().set(xlim=[-5, 5])
        plt.gca().set(title='gamma=' + str(g))
        plt.legend(fontsize=10)
    plt.show()
