from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from helpers.helper import PlotBoundaries
import numpy as np
import matplotlib.pyplot as plt

# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
#
# svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm.fit(X_train_std, y_train)
#
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
#
# bound_svm = PlotBoundaries(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105, 150))
# plt_svm = bound_svm.plot()
# plt_svm.xlabel('petal length [standardized]')
# plt_svm.ylabel('petal width [standardized]')
# plt_svm.legend(loc='upper left')
# plt_svm.show()

# create and display nonlinear separable dataset
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

# use SVM with Radial Basis Function Kernel to separate classes and plot decision boundaries
nonlin_svm = SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.2)
nonlin_svm.fit(X_xor, y_xor)
bound_nlsvm = PlotBoundaries(X=X_xor, y=y_xor, classifier=nonlin_svm)
plt_nlsvm = bound_nlsvm.plot()
plt_nlsvm.legend(loc='upper left')
plt_nlsvm.show()
