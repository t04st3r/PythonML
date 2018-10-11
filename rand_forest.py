from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from helpers.helper import PlotBoundaries, PlotCost
import numpy as np


forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=4)

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

forest.fit(X_train, y_train)

# rf_bound = PlotBoundaries(X=X_combined, y=y_combined, classifier=forest, test_idx=range(105, 150))
#
# plt = rf_bound.plot()
# plt.xlabel('petal length [cm]')
# plt.ylabel('sepal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

lr_predictions = forest.predict_proba(X_test[:, :])
lr_pred_results = np.c_[lr_predictions, y_test]
print(lr_pred_results)

errors = []

for x, y in zip(lr_predictions, y_test):
    x_list = x.tolist()
    errors.append(x_list.index(max(x_list)) != y)

#count number of misclassification
print('Total misclassifications on test set:', sum(errors))



