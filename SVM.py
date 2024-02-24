import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy.linalg import eig
from sklearn import svm

mnist = load_digits()
x = mnist.data
y = mnist.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=None)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

means = np.mean(x_train, axis=0)
A = x_train - means
C = np.cov(A, rowvar=False)
Eigenvalues, Eigenvectors = eig(C)
indicies = np.argsort(Eigenvalues)[::-1]
vectorssorted = Eigenvectors[:, indicies]
valuessorted = Eigenvalues[indicies]
eigen95 = np.sum(valuessorted) * 0.95
eigen90 = np.sum(valuessorted) * 0.90
variance = np.cumsum(valuessorted,axis=0)
components90 = np.argmax(np.cumsum(valuessorted) >= eigen90) + 1
components95 = np.argmax(np.cumsum(valuessorted) >= eigen95) + 1

x_train90 = np.dot(x_train, vectorssorted[:, :components90])
x_test90 = np.dot(x_test, vectorssorted[:, :components90])
x_train95 = np.dot(x_train, vectorssorted[:, :components95])
x_test95 = np.dot(x_test, vectorssorted[:, :components95])


clf = svm.SVC(gamma='scale',class_weight='balanced',C=100, kernel='rbf')
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
conf = confusion_matrix(predictions, y_test)
print("accuracy with C = 100: " + str(accuracy))
print(conf)
clf = svm.SVC(gamma='scale',class_weight='balanced',C=10, kernel='rbf')
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print("accuracy with C = 10: " + str(accuracy))
clf = svm.SVC(gamma='scale',class_weight='balanced',C=1, kernel='rbf')
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print("accuracy with C = 1: " + str(accuracy))
clf = svm.SVC(gamma='scale',class_weight='balanced',C=0.1, kernel='rbf')
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print("accuracy with C = 0.1: " + str(accuracy))
"""
PCA with 95% eigenvalues
"""
x_train = x_train95
x_test = x_test95


clf95 = svm.SVC(gamma='scale',class_weight='balanced',C=100, kernel='rbf')
clf95.fit(x_train, y_train)
predictions = clf95.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
conf = confusion_matrix(predictions, y_test)
print("")
print("PCA with 95% eigenvalues")
print(accuracy)
print(conf)

"""
PCA with 90% eigenvalues
"""
x_train = x_train90
x_test = x_test90

clf90 = svm.SVC(gamma='scale',class_weight='balanced',C=100, kernel='rbf')
clf90.fit(x_train, y_train)
predictions = clf90.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
conf = confusion_matrix(predictions, y_test)
print("")
print("PCA with 90% eigenvalues")
print(accuracy)
print(conf)