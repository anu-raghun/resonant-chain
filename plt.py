import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-2*np.pi,0,500)
plt.plot(x,x*np.cos(x),label='xcosx')
plt.plot(x,-x,label='-x')
plt.axvline(-1*np.pi,label='-$\pi$',color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Problem 1')
plt.legend()
plt.show()
