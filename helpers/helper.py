from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


class PlotBoundaries(object):
    """ Plot decision regions for a 2D dataset"""

    def __init__(self, X, y, classifier, resolution=0.02):
        self.X = X
        self.y = y
        self.classifier = classifier
        self.resolution = resolution

    def plot(self):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])

        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, self.resolution),
                               np.arange(x2_min, x2_max, self.resolution))

        Z = self.classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x=self.X[self.y == cl, 0], y=self.X[self.y == cl, 1],
                        alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        return plt


class PlotCost(object):
    """ Plot cost function through epochs"""

    def __init__(self, cost):
        self.cost = cost

    def plot(self):
        plt.plot(range(1, len(self.cost) + 1), self.cost, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        return plt
