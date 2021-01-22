from scipy import special
from numpy import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



mu=0
N=100

sd=1/np.sqrt(N)
x = random.normal(loc=mu, scale=sd, size=(N))
percentile95=mu+((sd)*(np.sqrt(2)*special.erfcinv(2*0.05)))
sns.distplot(x, hist=False)
plt.axvline(percentile95)
plt.show()

print(-np.sqrt(2)*special.erfcinv(2*0.05))