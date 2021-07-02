import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import *
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def createClusteredData(N, k): #creating fake data
    np.random.seed(0)
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(10000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 1000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y


X, y = createClusteredData(100, 5)

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X) #scaling data in range -1 to 1 for SVC
X = scaling.transform(X)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))
# plt.show()

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y) #support vector classifier

nx, ny = (3, 2)

x = np.linspace(0, 1, nx)

yy = np.linspace(0, 1, ny)

print(x)

print(yy)

xxxx= np.meshgrid(x, yy, np.linspace(0, 1, 3))

print(xxxx)

print(np.c_[np.array([1,2,3]), np.array([10, 14, 16]), np.array([4,5,6])])