from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers.helper import PlotBoundaries
import numpy as np
import sklearn.neighbors

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#get all possible distances
distances = sorted(sklearn.neighbors.VALID_METRICS['brute'])
print(distances)

distance = 'euclidean'  # 'manhattan'

p = 2 if distance == 'euclidean' else 1

knn = KNeighborsClassifier(n_neighbors=5, p=p, metric='minkowski')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

knn.fit(X_train_std, y_train)

knn_bound = PlotBoundaries(X=X_combined_std, y=y_combined, classifier=knn, test_idx=range(105, 150))

plt = knn_bound.plot()
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.suptitle(distance, fontsize=20)
plt.legend(loc='upper left')
plt.show()


