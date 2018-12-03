import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

NUMBER_OF_DIMENSIONS = 2

def fitness(data):
  print("in fitness", data)
  temp = 0.0
  sin_temp = 0.0
  fitness = 0.0
  for i in range(0, NUMBER_OF_DIMENSIONS):
      temp += (data[i] + ((-1)**(i+1))*((i+1)%4))**2
      sin_temp += data[i]**(i+1)
  fitness = -math.sqrt(temp) + math.sin(sin_temp)
  return fitness

X = np.linspace(-8, 8, 100)
print("X.shape: ", X.shape)
#Y = np.linspace(-8, 8, 100)
#print("Y.shape: ", Y.shape)
X = np.meshgrid(X, X)

print("X.shape: ", np.array(X).shape)

inSqrt = []
inSin = []
for dim in range(2):
  inSqrt.append((X[dim]+((-1)**(dim+1)*(dim+1)%4)**2))
  inSin.append(X[dim]**(dim+1))
print("list in sqrt", inSqrt)
print(inSin)


#Z = -np.sqrt((X-1)**2 + (Y+2)**2) + np.sin(X + Y**2)
Y = -np.sqrt(inSqrt) + np.sin(inSin)
print("Z.shape: ", Z.shape)
print(fitness(np.amax(Z, axis=-1)))
print("Z is", list(Z))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X[0], X[1], Y, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)
# plt.savefig('rastrigin_graph.png')
plt.show()

