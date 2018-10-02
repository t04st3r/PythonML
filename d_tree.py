from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from helpers.helper import PlotBoundaries
import numpy as np

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


dt_bound = PlotBoundaries(X=X_combined, y=y_combined, classifier=tree, test_idx=range(105, 150))
plt = dt_bound.plot()
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.show()

export_graphviz(tree, out_file='tree.dot', feature_names=['petal_length', 'sepal_length'])
