import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import *
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def createClusteredData(N, k): #creating fake data
    np.random.seed(12)
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

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y) #support vector classifier, C->regularization

def plotDemoPredictions(clf):
    #creates a cordinate matrix out of given cordinate vectors
    xx, yy = np.meshgrid(np.arange(-1.2, 1.2, .001),
                     np.arange(-1.2, 1.2, .001))
    
    # print(xx)
    # print(yy)
    #flattening to 1D array -> similar to np.reshape(-1)    
    npx = xx.ravel()
    npy = yy.ravel()
    
    # Convert to a list of 2D (income, age) points
    samplePoints = np.c_[npx, npy]
    
    # Generate predicted labels (cluster numbers) for each point
    Z = clf.predict(samplePoints)
    # print(Z)
    Z = Z.reshape(xx.shape) #Reshape results to match xx dimension
    # print(Z)
    plt.xlim(-1.2, 1.2)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5) # Draw the contour
    plt.scatter(X[:,0], X[:,1], c=y.astype(float)) # Draw the prev points
    plt.show()
    
plotDemoPredictions(svc)

print(svc.predict(scaling.transform([[200000, 70]])))
print(svc.predict(scaling.transform([[12000, 24]])))

""" 
nx, ny = (1, 2)

x = np.linspace(0, 1, nx)

yy = np.linspace(0, 1, ny)

print(x)

print(yy)

mesh= np.meshgrid(x, yy, np.linspace(0, 1, 3), indexing='ij') #indexing 'ij' -> (n1, n2, n3...) and 'xy'(default) -> (n2, n1, n3...)

print(mesh)

print(np.c_[np.array([1,2,3]), np.array([10, 14, 16]), np.array([4,5,6])])

"""

