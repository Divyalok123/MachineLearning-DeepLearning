import matplotlib
import numpy as np
from pylab import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
sns.set()

np.random.seed(0)
pageSpeeds = np.random.normal(5.0, 1.2, 1000)
purchaseAmount = np.random.normal(80.0, 10.0, 1000) / pageSpeeds

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

p4 = np.poly1d(np.polyfit(x, y, 4))
print(p4)

xp = np.linspace(0, 10, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='y')

r2 = metrics.r2_score(y, p4(x))
print(r2)

plt.show()