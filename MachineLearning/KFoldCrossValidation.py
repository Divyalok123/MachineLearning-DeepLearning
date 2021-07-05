import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn.datasets as ds
from sklearn import svm

iris = ds.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
classifier = svm.SVC(kernel="poly", degree=2, C=1).fit(X_train, y_train)
score1 = classifier.score(X_test, y_test)

print('Score with single train/test: ', score1)

scores = cross_val_score(classifier, iris.data, iris.target, cv=10)

print(scores)

score2 = scores.mean()

print('Score with kfold cross validation method: ', score2)




