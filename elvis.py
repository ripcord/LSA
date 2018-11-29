import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-8, 8, 100)
Y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(X, Y)

Z = -np.sqrt((X-1)**2 + (Y+2)**2) + np.sin(X + Y**2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)
# plt.savefig('rastrigin_graph.png')
plt.show()

