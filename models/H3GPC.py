__author__ = 'morgan'
import numpy as np
from H3Kernels import kernel


class H3GPC(object):
    """
    Gaussian process classification main class
    The implementation is referred to Rasmussen & Williams' GP book.
    Note: we use Laplace appriximation and logistic function for GPC.
    """

    def __init__(self, sigma=0.5, kernel='rbf', gamma=0.5,
                 coef0=1, degree=2):
        """
        object constructor
        :param
            sigma: noise level
            kernel: string type of kernel, {'rbf','linear','polynomial'}
            gamma: e.g., in rbf kernel: exp(-gamma|xj-xi|^2)
            coef0: for linear kernel, e.g., coef0*XX^T
            degree: for polynomial, e.g., |xi-xj|^(degree)
        :return:
        """
        self.sigma = sigma
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.likelihood_f = lambda x: 1. / (1 + np.exp(-x))

    def fit(self, X, y, Xt):
        """
        fit function of Gaussian process regression
        :param X: observations
        :param y: binary labels
        :param Xt: test samples
        :return: predictive class probability for Xt (for y=+1)
        """

        def laplace_mode(K, y):
            # find the mode of the Laplace appriximation
            f_new = f_old = np.zeros(n)
            while np.linalg.norm(f_new - f_old) > 1e-4:
                class_prob = self.likelihood_f(f_old)       # class=1 probability
                log_prime = .5 * (y + 1) - class_prob.ravel()
                W = np.diag((class_prob * (1 - class_prob)).ravel())
                W_root = np.sqrt(W)
                B = np.eye(n) + W_root.dot(K).dot(W_root)
                B_inv = np.linalg.inv(B)
                f_new = (K - K.dot(W_root).dot(B_inv).dot(W_root).dot(K)
                         ).dot(W.dot(f_old.reshape(n, 1)) + log_prime)

            return f_new

        n, d = X.shape
        K = kernel(X, metric=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree,
                   filter_params=True) + 1e-3
        K_xt = kernel(X, Xt, metric=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree,
                      filter_params=True)
        K_tt = kernel(Xt, metric=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree,
                      filter_params=True)
        f_mode = laplace_mode(K, y)
        class_prob = self.likelihood_f(f_mode).reshape(n, 1)
        W = np.diag((class_prob * (1 - class_prob)).ravel())
        W_root = np.sqrt(W)
        B = np.eye(n) + W_root.dot(K).dot(W_root)
        B_inv = np.linalg.inv(B)
        ft_mean = K_xt.T.dot(class_prob)
        ft_var = K_tt - K_xt.T.dot(W_root.dot(B_inv).dot(W_root)).dot(K_xt)
        # TODO: integral on class probability

        return ft_mean, ft_var


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X1 = np.random.multivariate_normal([-2.5, 2.5], [[0.5, 0], [0, 1.5]], 50)
    y1 = np.ones(X1.shape[0])
    X2 = np.random.multivariate_normal([2.5, 2.5], [[1.5, 0], [0, 0.5]], 50)
    y2 = -np.ones(X2.shape[0])
    X, y = np.r_[X1, X2], np.r_[y1, y2]
    X1t = np.random.multivariate_normal([-2.5, 2.5], [[0.5, 0], [0, 1.5]], 10)
    y1t = np.ones(X1t.shape[0])
    X2t = np.random.multivariate_normal([2.5, 2.5], [[1.5, 0], [0, 0.5]], 10)
    y2t = -np.ones(X2t.shape[0])
    Xt, yt = np.r_[X1t, X2t], np.r_[y1t, y2t]
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'ro')
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'bo')
    clf = H3GPC(gamma=0.5)
    ft_mean, ft_var = clf.fit(X, y, Xt)
    print ft_mean, ft_var
    plt.show()
