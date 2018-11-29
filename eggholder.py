import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-512, 512, 100)
Y = np.linspace(-512, 512, 100)
X, Y = np.meshgrid(X, Y)

Z = -(-(Y + 47) * np.sin(np.sqrt(abs(Y+X/2+47))) + (-X * np.sin(np.sqrt(abs(X-(Y+47))))))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)
# plt.savefig('rastrigin_graph.png')
plt.show()

