#finding the right degree polynomial in polynomial regression to fit the given set of data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
sns.set()
np.random.seed(0)

datax = np.random.normal(10, 2, 100);
datay = np.random.normal(30, 2, 100)/datax;

# to train 
trainx = datax[:80]
trainy = datay[:80]
trainx = np.array(trainx)
trainy = np.array(trainy)
# to test
testx = datax[80:]
testy = datay[80:]
testx = np.array(testx)
testy = np.array(testy)

#fit the data
p4 = np.poly1d(np.polyfit(trainx, trainy, 4));

plt.figure()
axes = plt.axes()
axes.set_xlim([7, 13])
axes.set_ylim([1, 5])
linedatax = np.linspace(7, 13, 100)
linedatay = p4(linedatax);
plt.scatter(trainx, trainy)
plt.plot(linedatax, linedatay, c="y")


plt.figure()
axes = plt.axes()
axes.set_xlim([7, 13])
axes.set_ylim([1, 5])
linedatax = np.linspace(7, 13, 100)
linedatay = p4(linedatax);
plt.scatter(testx, testy)
plt.plot(linedatax, linedatay, c="y")

print(metrics.r2_score(trainy, p4(trainx)))
print(metrics.r2_score(testy, p4(testx)))

plt.show()