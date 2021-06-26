import matplotlib
import numpy as np
from pylab import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set()

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(10, 3, 1000)) * 3

print('covar: ', np.cov(pageSpeeds, purchaseAmount))
print('correlation: ', np.corrcoef(pageSpeeds, purchaseAmount))

slope, intercept, rvalue, pvalue, stderr = stats.linregress(pageSpeeds, purchaseAmount)
print(slope, intercept, rvalue, pvalue, stderr)

#rsquared value
print(rvalue ** 2)

def getyval(x):
    return x * slope + intercept

fitLineForGivenPageSpeeds = getyval(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLineForGivenPageSpeeds, c='y')
plt.show()
