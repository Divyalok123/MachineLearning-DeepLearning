#predicting car prices

from numpy.lib import apply_over_axes
import pandas as pd
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set()

ourdata = pd.read_excel('../assets/CarsData.xls')
# pd.set_option("max_rows", None)
# pd.set_option('max_columns', None)

refData = ourdata[['Mileage', 'Price']]

bins = np.arange(0, 50000, 10000)
newRefData = pd.cut(refData['Mileage'], bins)

# print(newRefData.head(20))
groups = refData.groupby(newRefData)

# print(groups.groups)
# print(groups.head(30))

groupmeans = groups.mean()
# print(groupmeans.head())

# fig,axes = plt.subplots()
# fig.tight_layout(pad=5)
# groupmeans['Price'].plot()s
# plt.xticks(rotation=45)

# plt.figure()
# groups['Mileage'].plot()
# plt.figure()
# groups['Price'].plot()

scale = StandardScaler()

X = ourdata[['Mileage', 'Cylinder', 'Doors']]
y = ourdata['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].values)
X = sm.add_constant(X)

est = sm.OLS(y, X).fit()

print(est.params)
print(est.summary())

ygroup = y.groupby(ourdata.Doors).mean()
print(ygroup)

scaled = scale.transform([[31000, 6, 2]])
scaled = np.insert(scaled[0], 0, 1)
print(scaled)

prediction = est.predict(scaled)
print(prediction)

scaled = scale.transform([[45000, 4, 4]])
scaled = np.insert(scaled[0], 0, 1)
print(scaled)

prediction = est.predict(scaled)
print(prediction)

scaled = scale.transform([[1000, 8, 2]])
scaled = np.insert(scaled[0], 0, 1)
print(scaled)

prediction = est.predict(scaled)
print(prediction)

scaled = scale.transform([[40000, 7, 4]])
scaled = np.insert(scaled[0], 0, 1)
print(scaled)

prediction = est.predict(scaled)
print(prediction)
# plt.show()

