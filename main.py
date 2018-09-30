import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from estimators.est import Perceptron, AdalineGD, AdalineSGD
from helpers.helper import PlotBoundaries
import os

file = 'iris.data'
if not os.path.exists(file):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.to_csv(file, index=False, header=None)
else:
    df = pd.read_csv(file, header=None)

print('shape', df.shape)
print(df.describe())
print(df.head())
print('....')
print(df.tail())
print('classes count:', df[4].value_counts())

# extract two classes
y = df.iloc[0:100, 4].values

# encode two classes as 1 and -1
y = np.where(y == 'Iris-setosa', -1, 1)

# extract input features
X = df.iloc[0:100, [0, 2]].values

#Normalization through standardization

# for the jth sample: x_jstd = (x_j - average_j) / std_deviation_j

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# plot two classes scatterplot
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('petal length')
# plt.ylabel('sepal length')
# plt.legend(loc='upper left')
# plt.show()

# plot misclassifications through epocs graph
# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('# of Misclassifications')
# plt.show()

#plot decision boundaries
# bound = PlotBoundaries(X, y, classifier=ppn)
# bound_plt = bound.plot()
# bound_plt.xlabel('sepal length [cm]')
# bound_plt.ylabel('petal length [cm')
# bound_plt.legend(loc='upper left')
# bound_plt.show()

#plot Adaline misclassifications through epocs with different learning rates
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

#first subgraph
# ada1 = AdalineGD(n_iter=30, eta=0.01).fit(X_std, y)
# ax[0].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Sum Squared Error')
# ax[0].set_title('Adaline - Learning rate 0.01')

#second subgraph
# ada2 = AdalineGD(n_iter=30, eta=0.0001).fit(X_std, y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Sum Squared Error')
# ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.show()

#plot decision boundaries for Adaline
# bound = PlotBoundaries(X_std, y, classifier=ada1)
# bound_plt = bound.plot()
# bound_plt.xlabel('sepal length [cm]')
# bound_plt.ylabel('petal length [cm')
# bound_plt.legend(loc='upper left')
# bound_plt.show()

#plot SGD misclassifications through Epochs
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

#plot decision boundaries for SGD
bound = PlotBoundaries(X_std, y, classifier=ada)
bound_plt = bound.plot()
bound_plt.xlabel('sepal length [cm]')
bound_plt.ylabel('petal length [cm')
bound_plt.legend(loc='upper left')
bound_plt.show()
