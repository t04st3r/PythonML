import numpy as np


class Perceptron(object):
    """Perceptron classifier

    Params
    ------

    eta: float Learning rate (between 0.0 and 1.0)
    n_iter: int passes over the training datasets

    Attributes
    ----------

    w_ : 1d-array Weights after fitting
    errors_ : list # of misclassifications in every epoch

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """

        :param X: array Training vector shape = [n_samples, n_features]
        :param y: array Target values shape = [n_samples]
        :return: self object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineGD(object):
    """ADAptive LInear NEuron classifier (aka Batch Gradient Descent)

    Params
    ------

    eta: float Learning rate (between 0.0 and 1.0)
    n_iter: int passes over the training dataset

    Attributes
    ----------

    w_ : 1d-array Weights after fitting
    cost_ : list # of misclassifications in every epoch

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data

        :param X: array Training vector shape = [n_samples, n_features]
        :param y: array Target values shape = [n_samples]
        :return: self object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier (aka Stochastic Gradient Descent)

    Params
    ------

    eta: float Learning rate (between 0.0 and 1.0)
    n_iter: int passes over the training dataset

    Attributes
    ----------

    w_ : 1d-array Weights after fitting
    cost_ : list # of misclassifications in every epoch
    shuffle : bool (default: True) if True prevent from cycles
    random_state : int (default: None) Set random state for shuffling and initializing the weights
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """Fit training data

        :param X: array Training vector shape = [n_samples, n_features]
        :param y: array Target values shape = [n_samples]
        :return: self object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle :
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_filt(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized :
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
