import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from numpy import random
from numpy.core.numeric import cross
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from IPython.display import Image  
import pydotplus
from six import StringIO  
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import RMSprop

file_path = 'data/mammographic_masses.csv'
all_features = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']

### preparing the data

data = pd.read_csv(file_path, na_values=['?'], names=all_features)

# checkdata = data.loc[(data['age'].isnull()) | (data['shape'].isnull()) | (data['margin'].isnull()) | (data['density'].isnull())]
# print(checkdata.shape)
# print(checkdata.head(10))

data.dropna(inplace=True)
features = all_features[1:5]
features_data = data[features].values
features_data = StandardScaler().fit_transform(features_data)
target_classes = data['severity'].values

### --- USING DECISION TREES ---

"""
## splitting the data
np.random.seed(0)
(train_X, test_X, train_y, test_y) = train_test_split(features_data, target_classes, test_size=0.25, random_state=1)

# print(train_X.shape)
# print(test_X.shape)

## getting and applying the classifier
clf = tree.DecisionTreeClassifier(random_state=1)
clf.fit(train_X, train_y)

## printing the decision tree 
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
# graph.write_png('results/decision_tree.jpg')

# score of this single train test
score = clf.score(test_X, test_y)
print(score) # 73.56%

# k-fold cross validation
score = cross_val_score(clf, features_data, target_classes, cv=20)
print(score.mean()) # 77.10%

## using random forest classfier
clf = RFC(n_estimators=20)
clf = clf.fit(train_X, train_y)

score = cross_val_score(clf, features_data, target_classes, cv=10)
print(score.mean()) #76.89

"""

### --- USING SUPPORT VECTOR MACHINES ---

"""
C = 1.0
svc = svm.SVC(kernel='rbf', degree=2, C=C)
score = cross_val_score(svc, features_data, target_classes, cv = 12)
print(score.mean()) #80% -> performed best out of all kernels
"""

### --- USING K Nearest Neighbors ---

"""
# KNN = KNeighborsClassifier(n_neighbors=13)
# score = cross_val_score(KNN, features_data, target_classes, cv=12)
# print(score.mean()) #79.40

maxscorewithKNN = 0
numneighbors = 0
for i in range(5, 50):
    KNN = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(KNN, features_data, target_classes, cv=24)
    print('Score with ', i, ' neighbors: ', score.mean())
    if score.mean() > maxscorewithKNN:
        numneighbors = i
        maxscorewithKNN = score.mean()

print(maxscorewithKNN, ' for ', numneighbors, ' neighbors') #80.11% - 15 neighbors - with cv=24

"""

### --- USING NAIVE BAYES CLASSIFIERS --- 

"""
features_data = MinMaxScaler().fit_transform(data[features].values)
clf = naive_bayes.MultinomialNB()
score = cross_val_score(clf, features_data, target_classes, cv=10)
print(score.mean()) # 78.44% accuracy
"""

### --- USING LOGISTIC REGRESSION ---

#in its basic form uses a logistic function to model a binary dependent variable

"""
(train_X, test_X, train_y, test_y) = train_test_split(features_data, target_classes, test_size=0.3, random_state=1)
clf = LogisticRegression(random_state=0)
clf = clf.fit(train_X, train_y)
score = clf.score(test_X, test_y)
print(score) # 79.51% accuracy

score = cross_val_score(clf, features_data, target_classes, cv=10)
print(score.mean()) # 80.72% accuracy
"""

### --- USING NEURAL NETWORK ---

def makeModel():
    model = Sequential()
    model.add(Dense(6, input_dim=4, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=makeModel, epochs=100, verbose=0)
score = cross_val_score(estimator, features_data, target_classes, cv=10)
print(score.mean()) # 80.36% accuracy