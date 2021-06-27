from numpy import random, array
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random
import seaborn as sns
sns.set();


def createClusteredData(N, k): #creating fake data
    random.seed(0)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(10000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 1000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    print(X)
    return X

data = createClusteredData(100, 5)

model = KMeans(n_clusters=5)

model = model.fit(scale(data))

print(model.labels_)

plt.figure(figsize=(10, 5))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()