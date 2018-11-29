import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-100, 100, 200)
Y = np.linspace(-100, 100, 200)
X, Y = np.meshgrid(X, Y)

A = np.cos(np.sin(np.abs(np.power(X,2) - np.power(Y,2)))) - 0.5
B = np.power((1 + 0.001*(np.power(X,2) + np.power(Y,2))),2)
Z = 0.5 + A/B

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)
# plt.savefig('rastrigin_graph.png')
plt.show()                                    
