import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
def plot_decision_boundary(classifier, X, y):
 x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
 np.arange(y_min, y_max, 0.1))
 Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)
 plt.contourf(xx, yy, Z, alpha=0.4)
 plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
 plt.xlabel('Feature 1')
 plt.ylabel('Feature 2')
 plt.title('SVM Decision Boundary')
 plt.show()
plot_decision_boundary(svm_classifier, X_train, y_train) 