#principle component analysis

from numpy import load
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

iris = load_iris()

numSamples, numFeatures = iris.data.shape
# print(iris.target_names)
# print(iris.target)
# print(len(iris.target))
# print(numSamples, numFeatures)

data = iris.data

# print(data[:10])

pca = PCA(n_components=2, whiten=True).fit(data)
data_pca = pca.transform(data)

# print(data_pca[:10])


# print(pca.explained_variance_ratio_)
# print(sum(pca.explained_variance_ratio_))

colors = 'rgb'
target_ids = range(len(iris.target_names))
plt.figure()

for id, color, label in zip(target_ids, colors, iris.target_names):
    plt.scatter(data_pca[iris.target == id, 0], data_pca[iris.target == id, 1], c = color, label=label)

plt.legend()
plt.show()
