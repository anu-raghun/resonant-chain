from IPython.display import display
import sympy as sy
from sympy.solvers.ode import dsolve
import matplotlib.pyplot as plt
import numpy as np

sy.init_printing()  # LaTeX like pretty printing for IPython


t = sy.Symbol('t')
f = sy.Function('f')

T1=((7/6)*t)+300
T3=300
R23=211.64
R21=3315.73
eq2=4.17*(10**(-15))*((f(t)**4)*(R23+R21))-((R23*(T3**4))+((R21*(T1**4))))
eq1 = sy.Eq(f(t).diff(t), eq2)  # the equation 
sls = dsolve(eq1)  # solvde ODE

# print solutions:
print("For ode")
display(eq1)
print("the solutions are:")
display(sls)

# plot solutions:
x = np.linspace(0, 2, 100)
lam_x = sy.lambdify(t, sls, modules=['numpy'])

x_vals = np.linspace(0, 10, 100)
y_vals = lam_x(x_vals)

plt.plot(x_vals, y_vals)
plt.ylabel("Speed")
plt.show()
