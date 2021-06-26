import matplotlib
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.rand(100)

#moments
print(np.mean(incomes))
print(np.var(incomes))
print(sts.skew(incomes))
print(sts.kurtosis(incomes))

# #plot
plt.hist(incomes)
plt.show()
