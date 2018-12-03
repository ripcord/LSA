from __future__ import print_function

import math
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def plot(input, output, dimensions):
    #xAxis = np.linspace(-xRange, xRange, 10**2)
    #yAxis = np.linspace(-yRange, yRange, 10**2)
    print(input.shape)
    columnTitles = []
    #data = np.zeros((dimensions, input.shape[0], 1))
    data = np.meshgrid(input, input)
    output = 
    print("data.shape", np.array(data).shape)
    print(list(data))
    #inputShape = input.shape
    #for dim in range(dimensions):
    #    data[dim] = input
    #    data[dim][0] = -np.sqrt((input-1)**2 + (input+2)**2) + np.sin(input + input**2)
    #data = np.reshape(data, -1))
    print(np.array(data).shape)
    print(list(data))
    for dim in range(dimensions):
        columnTitles.append("dimension: " + str(dim + 1))
    print(columnTitles)
    data = pd.DataFrame(input, columns=columnTitles)
    print(data.shape)
    #print(data.shape)
    #print(data)
    scatter_matrix(data, alpha=0.2, figsize=(100, 100), diagonal='kde')
    #plt.figure()
    #data.plot()
    #plt.show()



    #Z = -np.sqrt((input-1)**2 + (input+2)**2) + np.sin(input + input**2)

    #print(xAxis)

    #input, output = np.meshgrid(input, input)
    #print("input.shape", input.shape)
    #print("output.shape", output.shape)

    #fig = plt.figure() 
    #ax = fig.gca(projection='3d') 
    #ax.plot_surface(input, input, Z, rstride=1, cstride=1,
    #cmap=cm.nipy_spectral, linewidth=0.08,
    #antialiased=True)    
    ## plt.savefig('rastrigin_graph.png')
    #plt.show()
