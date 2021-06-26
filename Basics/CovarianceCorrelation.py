import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()
plt.xkcd()

x = np.random.normal(100, 10, 1000)

# y = np.random.normal(100, 10, 1000) #random -> no correlation
y = 100 + x*10 #Now there is perfect positive correlation
z = 100 - x*10 #There is perfect negative correlation b/w x and z

plt.figure()
plt.scatter(x, y)
plt.figure()
plt.scatter(x, z)

a = [1, 2, 3]
b = [4, 5, 7]

print(np.dot(a, b))

print(np.cov(x,y))
print(np.corrcoef(x,y))
print(np.cov(x,z))
print(np.corrcoef(x,z))

plt.show()
