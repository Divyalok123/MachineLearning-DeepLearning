import numpy as np
import scipy.stats as sts

A = np.random.normal(25.0, 5.0, 10000)
B = np.random.normal(26.0, 5.0, 10000)

TP = sts.ttest_ind(A, B)

print(TP)

A = np.random.normal(25, 1, 10000)
B = np.random.normal(25, 1, 10000)

TP = sts.ttest_ind(A, B)

print(TP)

A = np.random.normal(25, 3, 50000)
B = np.random.normal(26, 3, 50000)

TP = sts.ttest_ind(A, B)

print(TP)

A = np.random.normal(25, 10, 70000)
B = np.random.normal(26, 10, 70000)

TP = sts.ttest_ind(A, B)

print(TP)
