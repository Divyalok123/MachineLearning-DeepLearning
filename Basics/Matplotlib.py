import numpy as np
import scipy.stats as sts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ourfile = pd.read_csv('../assets/FuelEfficiency.csv')
sns.set(rc={'figure.figsize':(15, 10)})
gg = ourfile.pivot_table(index='# Gears', columns='Eng Displ', values='CombMPG', aggfunc='mean')
g = sns.heatmap(gg)
plt.xticks(rotation=45)
plt.show()