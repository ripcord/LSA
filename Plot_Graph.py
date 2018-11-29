import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot(function, dimensions, xRange, yRange):
    xAxis = np.linspace(-xRange, xRange, 10**2)
    yAxis = np.linspace(-yRange, yRange, 10**2)

    print(xAxis)

    xAxis, yAxis = np.meshgrid(xAxis, yAxis)

    fig = plt.figure() 
    ax = fig.gca(projection='3d') 
    ax.plot_surface(xAxis, yAxis, function, rstride=1, cstride=1,
    cmap=cm.nipy_spectral, linewidth=0.08,
    antialiased=True)    
    # plt.savefig('rastrigin_graph.png')
    plt.show()
