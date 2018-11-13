from __future__ import print_function

import sys
import math
import numpy as np
import random
import time
import copy

NUMBER_OF_POINTS = None
NUMBER_OF_DIMENSIONS = None
DATASET = None
DIMENSIONAL_MIDPOINT = None
ELITE_DATASET = None
BOUNDS = (-8, 8)
MAX_MOVEMENT_AMOUNT = 0.01 #FIXED FOR NOW

#fitness function, elvis needs boats
def fitness(data):
    temp = 0.0
    sin_temp = 0.0
    fitness = 0.0
    for i in range(0, NUMBER_OF_DIMENSIONS):
        temp += (data[i] + ((-1)**(i+1))*((i+1)%4))**2
        sin_temp += data[i]**(i+1)
    fitness = -math.sqrt(temp) + math.sin(sin_temp)
    return fitness

#get arguments from cmd, they go as follows {arg1: X points, arg2: N dimensions}
#X points on N dimensions
if(len(sys.argv) == 3 ):
    try:
        sys.argv[1] = int(sys.argv[1])
        sys.argv[2] = int(sys.argv[2])
    except ValueError:
        print("Arguments must be of type integer")
        exit()
    NUMBER_OF_DIMENSIONS = sys.argv[1]
    NUMBER_OF_POINTS = sys.argv[2]
else:
    print("Invalid Arguments")
    exit()

#CREATION OF THE ARRAY OF X POINTS ON N DIMENSIONS => DATASET
DATASET = np.full((NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS), None)

for point , i in enumerate(DATASET):
    for dim, x in enumerate(i):
        DATASET[point][dim] = random.uniform(BOUNDS[0], BOUNDS[1])
print("--*FULL DATASET BEFORE:\n",DATASET)

runs = 0
#Meat of the program, Merge/Fitness Function portion of the program
while True:
    #Determine best dataset => ELITE_DATASET
    # 1 point per dimension. there can only be 1 best
    ELITE_DATASET = DATASET[0]
    for i in DATASET:
        #print(i, fitness(i))
        if(fitness(i) > fitness(ELITE_DATASET)):
            ELITE_DATASET = i[:]

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
        if(DATASET[point].all() == ELITE_DATASET.all()):
            #run a search algorithm
            #print("EQUAL")
            time.sleep(.00001)
        else:
            for i in dimensions_to_change:
                if(x[i] < DIMENSIONAL_MIDPOINT[i]):
                    DATASET[point][i] += random.uniform(0, MAX_MOVEMENT_AMOUNT)
                if(x[i] > DIMENSIONAL_MIDPOINT[i]):
                    DATASET[point][i] -= random.uniform(0, MAX_MOVEMENT_AMOUNT)
        #print(DATASET)

    runs += 1    
    #break condition
    if runs > 10000:
        break

print("--*FULL DATASET AFTER:\n", DATASET)
print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))

