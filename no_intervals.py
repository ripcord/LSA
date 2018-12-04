from __future__ import print_function
from __future__ import division

import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import random
import time
import copy
import timeit

ELVIS_NEEDS_BOATS = "ELVIS"
SCHAFFER = "SCHAFFER"
RASTRIGIN = "RASTRIGIN"
EGGHOLDDER = "EGGHOLDER"

def getOutput(name, leftBound, rightBound):
    pointsOnLine = 10**2
    if name == ELVIS_NEEDS_BOATS:
        X = np.linspace(leftBound, rightBound, pointsOnLine)
        Y = X
        X, Y = np.meshgrid(X, Y)
        Z = -np.sqrt((X-1)**2 + (Y+2)**2) + np.sin(X + Y**2)
        return X, Y, Z
    elif name == SCHAFFER:
        raise ValueError("Not implemented yet")
    elif name == RASTRIGIN:
        raise ValueError("Not implemented yet")
    elif name == EGGHOLDDER:
        raise ValueError("Not implemented yet")
    else:
        raise ValueError("Function not known")
        


NUMBER_OF_POINTS = None         #Number of points
NUMBER_OF_DIMENSIONS = None     #Number of dimensions
DATASET = None                  #Set of all points and their given positions
DIMENSIONAL_MIDPOINT = None     #Set of all midpoints for each dimension
ELITE_DATASET = None            #The elite point, AKA the point with the best fitness
BOUNDS = (-8, 8)                #The global search interval
STEP_SIZE = None                #Step size for all points
ELITE_STEP_SIZE = None          #Step size for elite point
ITERS = 1000                    #Number of iterations
ELITE_CLIMBS = 20               #Number of times to hill climb for the elite point
FITNESS_EVALS = None            #Number of fitness evaluations (optional)

#Fitness function "Elvis Needs Boats"
#2D optimum = ~0.41

def elvis_needs_boats(data):
    temp = 0.0
    sin_temp = 0.0
    fitness = 0.0
    for i in range(0, NUMBER_OF_DIMENSIONS):
        temp += (data[i] + ((-1)**(i+1))*((i+1)%4))**2
        sin_temp += data[i]**(i+1)
    fitness = -math.sqrt(temp) + math.sin(sin_temp)
    return fitness

def fitness(data):
    return elvis_needs_boats(data)


#Get arguments from cmd, they go as follows {arg1: X points, arg2: N dimensions}
#X points on N dimensions
if (len(sys.argv) == 3 ):
    try:
        sys.argv[1] = int(sys.argv[1])
        sys.argv[2] = int(sys.argv[2])
    except ValueError:
        print("Arguments must be of type integer")
        exit()
    NUMBER_OF_DIMENSIONS = sys.argv[1]
    NUMBER_OF_POINTS = sys.argv[2]
elif (len(sys.argv) == 4):
    try:
    	sys.argv[1] = int(sys.argv[1])
    	sys.argv[2] = int(sys.argv[2])
    	sys.argv[3] = int(sys.argv[3])
    except ValueError:
    	print("Arguments must be of type integer")
    	exit()
    NUMBER_OF_DIMENSIONS = sys.argv[1]
    NUMBER_OF_POINTS = sys.argv[2]
    FITNESS_EVALS = sys.argv[3]
else:
    print("Usage: \"python", sys.argv[0], "[NUMBER OF DIMENSIONS] [NUMBER OF POINTS]\"")
    print("OR\nUsage: \"python", sys.argv[0], "[NUMBER OF DIMENSIONS] [NUMBER OF POINTS] [NUMBER OF FITNESS EVALUATIONS]\"")
    exit()


#ELITE_STEP_SIZE= 1/math.log(NUMBER_OF_DIMENSIONS)
#STEP_SIZE = 1/math.log(NUMBER_OF_DIMENSIONS)
#print(STEP_SIZE)
#print(ELITE_STEP_SIZE)

#PLOT = False
PLOT = True

# CHANGE FOR DIFFERENT FUNCTIONS
functionName = ELVIS_NEEDS_BOATS



if PLOT and NUMBER_OF_DIMENSIONS == 2 and functionName == ELVIS_NEEDS_BOATS:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(60,35)
    '''
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[0], BOUNDS[1])
    ax.set_zlim(BOUNDS[0], BOUNDS[1])
    '''


if not FITNESS_EVALS:
    FITNESS_EVALS = 1

evals = 0

while evals < FITNESS_EVALS:
    #CREATION OF THE ARRAY OF X POINTS ON N DIMENSIONS => DATASET
    #Creating the "followers"
    DATASET = np.full((NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS), None)
    datasetFitness = np.full((NUMBER_OF_POINTS), None)
    #print(datasetFitness)

    # for INDEX, VALUE in ITERABLE:
    for point , i in enumerate(DATASET):
        for dim, x in enumerate(i):
            DATASET[point][dim] = random.uniform(BOUNDS[0], BOUNDS[1])
    #print("dataset initialized", DATASET)
    #print(DATASET)
    for i in range(len(DATASET)):
        #print(i)
        #print(DATASET[i])
        #print("Dataset", DATASET[i])
        datasetFitness[i] = fitness(DATASET[i])
       
    #Meat of the program, Merge/Fitness Function portion of the program
    #print(datasetFitness)
    runs = 1
    optimum_runs = 0

    ELITE_DATASET = DATASET[0]
#        ELITE_PREVIOUS = copy.deepcopy(ELITE_DATASET)
#        while True:
    start_time = timeit.default_timer()
    while runs <= ITERS:

        if PLOT and NUMBER_OF_DIMENSIONS == 2 and functionName == ELVIS_NEEDS_BOATS:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Fitness')
            
            x, y, z = getOutput(functionName, BOUNDS[0], BOUNDS[1])
            ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='gray', linewidth=0.08,
                antialiased=True)
            
            best = list(datasetFitness).index(np.amax(datasetFitness, axis=-1))
            ax.scatter3D(DATASET[best][0], DATASET[best][1], datasetFitness[best]+0.5, s=200, c='yellow',marker='D', alpha=1, edgecolor='white')
            for i in range(len(DATASET)):
                if i != best:
                    ax.scatter3D(DATASET[i][0], DATASET[i][1], datasetFitness[i]+0.5, s=200, c='orange',alpha=1, edgecolor='white')
            fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.05)

        ELITE_STEP_SIZE = ((0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
        STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]

        # Determine best dataset => ELITE_DATASET
        # 1 point per dimension. there can only be 1 best
        for index in range(len(DATASET)):
            i = DATASET[index]
            if fitness(i) > fitness(ELITE_DATASET):
                #print("best changed")
                ELITE_DATASET = copy.deepcopy(i[:])

        prev_fitness = fitness(ELITE_DATASET)

        #Find the midpoint of all the points
        DIMENSIONAL_MIDPOINT = np.full((NUMBER_OF_DIMENSIONS, 1), None)
        for index, i in enumerate(DATASET.T):
            DIMENSIONAL_MIDPOINT[index] = np.mean(i)
        #print("THE MIDPOINT", DIMENSIONAL_MIDPOINT)


        #random dimensions to change
        dimensions_to_change = random.sample(range(0, NUMBER_OF_DIMENSIONS), random.randint(0, NUMBER_OF_DIMENSIONS))
        #print(dimensions_to_change)
        
        # Move in those specific random dimensions toward middle
        for point, x in enumerate(DATASET):
            #print(point, ":DATASET:", DATASET[point], "\nELITE_DATASET", ELITE_DATASET)
            if fitness(DATASET[point]) >= fitness(ELITE_DATASET):
                ELITE_DATASET = copy.deepcopy(DATASET[point])

                #Hill climbing for the elite point
                for climb in range(ELITE_CLIMBS):
                    for i in range(NUMBER_OF_DIMENSIONS):
                        TEMP = copy.deepcopy(DATASET[point])
                        
                        DATASET[point][i] += random.uniform(0,ELITE_STEP_SIZE)
                                                
                        if not (fitness(DATASET[point]) > fitness(TEMP)):
                            for x in range(NUMBER_OF_DIMENSIONS):
                                DATASET[point][x] = TEMP[x]
                        
                            DATASET[point][i] -= random.uniform(0,ELITE_STEP_SIZE)
                            
                            if not (fitness(DATASET[point]) > fitness(TEMP)):
                                for x in range(NUMBER_OF_DIMENSIONS):
                                    DATASET[point][x] = TEMP[x]
                            #if(fitness(DATASET[point]) > fitness(ELITE_DATASET)):
                                #print("HILLCLIMBING => Old Best:", fitness(DATASET[point]), "New Best:", fitness(ELITE_DATASET)) 
                        ELITE_DATASET = copy.deepcopy(DATASET[point]) #CRITICAL LINE!
                        
            else:
                for i in dimensions_to_change:
                    if(x[i] < DIMENSIONAL_MIDPOINT[i]):
                        DATASET[point][i] += random.uniform(0, STEP_SIZE)
                    if(x[i] > DIMENSIONAL_MIDPOINT[i]):
                        DATASET[point][i] -= random.uniform(0, STEP_SIZE)

            # Update the fitness of all wolves
            for i in range(len(DATASET)):
                datasetFitness[i] = fitness(DATASET[i])

        runs += 1

        #if(prev_fitness < fitness(ELITE_DATASET)):
        #    print(runs, fitness(ELITE_DATASET))

        if (runs % 10) == 0:
            if (fitness(ELITE_DATASET) - prev_fitness) < 0.000009:
                #break
                optimum_runs = runs
                runtime = timeit.default_timer() - start_time
                runs = ITERS + 1
        #print(datasetFitness)

        #print(datasetFitness[best])

        #best = list(datasetFitness).index(np.amax(datasetFitness, axis=-1))
        #ax.scatter(DATASET[best][0], DATASET[best][1], datasetFitness[best], s=100, marker="D", c='purple',alpha=1, edgecolor='white')
        #for i in range(len(DATASET)):
        #    if i != best:
        #        ax.scatter(DATASET[i][0], DATASET[i][1], datasetFitness[i], s=100, c='red',alpha=1, edgecolor='white')
        #        plt.draw()
        #plt.draw()
        #plt.show()
        #best = list(datasetFitness).index(np.amax(datasetFitness, axis=-1))
        #print("best eval", fitness(ELITE_DATASET), "foundBest", datasetFitness[best])

    #plt.show()
    evals += 1

    #print("--*FULL DATASET AFTER:\n", DATASET)
    #print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
    #print("--*BEST DATA:", ELITE_DATASET, "{:>2}" .format(" "), "|","{:>2}" .format(" "),"FITNESS:", fitness(ELITE_DATASET))
    print("{s1:<{width}} {s2}".format(s1="--*BEST DATA: " + str(ELITE_DATASET), width=10 * ((NUMBER_OF_DIMENSIONS + 20)//2), s2="| FITNESS: " + str(fitness(ELITE_DATASET)) ))
    print("\tApproximate Runtime: {0:.5f} sec".format(runtime))
    print("\tRuns:", optimum_runs)
    #print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))
    #print("{s1} {s2:>{width}}".format(s1="--*BEST DATA: " + str(ELITE_DATASET), width=30, s2="| FITNESS: " + str(fitness(ELITE_DATASET)) ))

