from __future__ import print_function
from __future__ import division

import sys
import math
import numpy as np
import random
import time
import copy
import timeit

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


if not FITNESS_EVALS:
    FITNESS_EVALS = 1

evals = 0

while evals < FITNESS_EVALS:
    #CREATION OF THE ARRAY OF X POINTS ON N DIMENSIONS => DATASET
    #Creating the "followers"
    DATASET = np.full((NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS), None)


    # for INDEX, VALUE in ITERABLE:
    for point , i in enumerate(DATASET):
        for dim, x in enumerate(i):
            DATASET[point][dim] = random.uniform(BOUNDS[0], BOUNDS[1])
    #print("--*FULL DATASET BEFORE:\n",DATASET)


    #Meat of the program, Merge/Fitness Function portion of the program
    runs = 1
    optimum_runs = 0

    ELITE_DATASET = DATASET[0]
#        ELITE_PREVIOUS = copy.deepcopy(ELITE_DATASET)
#        while True:
    start_time = timeit.default_timer()
    while runs <= ITERS:
        ELITE_STEP_SIZE = ((0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
        STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]

        #Determine best dataset => ELITE_DATASET
        # 1 point per dimension. there can only be 1 best
        for i in DATASET:
            #print(i, fitness(i))
            if fitness(i) > fitness(ELITE_DATASET):
                #print("Old Best:", fitness(ELITE_DATASET), "New Best:", fitness(i))
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
        
        #move in those specific random dimensions toward middle
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
                        #DATASET[point][i] += STEP_SIZE
                    if(x[i] > DIMENSIONAL_MIDPOINT[i]):
                        DATASET[point][i] -= random.uniform(0, STEP_SIZE)
                        #DATASET[point][i] -= STEP_SIZE
            #print(DATASET)

        runs += 1

        #if(prev_fitness < fitness(ELITE_DATASET)):
        #    print(runs, fitness(ELITE_DATASET))

        if (runs % 10) == 0:
            if (fitness(ELITE_DATASET) - prev_fitness) < 0.000009:
                #break
                optimum_runs = runs
                runtime = timeit.default_timer() - start_time
                runs = ITERS + 1

    evals += 1

    #print("--*FULL DATASET AFTER:\n", DATASET)
    #print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
    #print("--*BEST DATA:", ELITE_DATASET, "{:>2}" .format(" "), "|","{:>2}" .format(" "),"FITNESS:", fitness(ELITE_DATASET))
    print("{s1:<{width}} {s2}".format(s1="--*BEST DATA: " + str(ELITE_DATASET), width=10 * ((NUMBER_OF_DIMENSIONS + 20)//2), s2="| FITNESS: " + str(fitness(ELITE_DATASET)) ))
    print("\tApproximate Runtime: {0:.5f} sec".format(runtime))
    print("\tRuns:", optimum_runs)
    #print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))
    #print("{s1} {s2:>{width}}".format(s1="--*BEST DATA: " + str(ELITE_DATASET), width=30, s2="| FITNESS: " + str(fitness(ELITE_DATASET)) ))

