from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
from helpers.helper import PlotBoundaries
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
n = (y_test != y_pred).sum()
errors = n / len(y_test)

print('Misclassified:', n)
print('Errors:', round(errors, 3))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

bound_perc = PlotBoundaries(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt_perc = bound_perc.plot()
plt_perc.xlabel('petal length [standardized]')
plt_perc.ylabel('petal width [standardized]')
plt_perc.legend(loc='upper left')
plt_perc.show()

#Logistic regression classifier
lr = LogisticRegression(C=1000.0, random_state=0, solver='lbfgs', multi_class='auto')
lr.fit(X_train_std, y_train)
bound_lr = PlotBoundaries(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
plt_lr = bound_lr.plot()
plt_lr.xlabel('petal length [standardized]')
plt_lr.ylabel('petal width [standardized]')
plt_lr.legend(loc='upper left')
plt_lr.show()

#format float on numpy array-ndarray
np.set_printoptions(precision=8, suppress=True)

#predict probability over test samples
lr_predictions = lr.predict_proba(X_test_std[:1, :])
print(lr_predictions)

