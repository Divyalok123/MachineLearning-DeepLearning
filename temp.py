import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

s = np.random.uniform(-1,0,1000)
import matplotlib.pyplot as plt

count, bins, ignored = plt.hist(s, 15, density=True)
print('count: ', count)
print('bins: ', bins)
print('ignored: ', ignored)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

plt.show()