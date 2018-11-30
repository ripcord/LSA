from __future__ import print_function
from __future__ import division

import sys
import math
import numpy as np
import random
import time
import copy

NUMBER_OF_POINTS = None         #Number of followers (wolves)
NUMBER_OF_DIMENSIONS = None     #Number of dimensions
DATASET = None                  #Set of all followers and their given positions
DIMENSIONAL_MIDPOINT = None     #Set of all midpoints for each dimension
ELITE_DATASET = None            #The best follower
BOUNDS = (-8, 8)                #The glcobal search interval
MAX_MOVEMENT_AMOUNT = None #FIXED FOR NOW
BEST_MOVEMENT_AMOUNT = None

#Fitness function (Elvis needs Boats)
#Elvis optimum = 0.41 (2 dimensions)
def fitness(data):
    temp = 0.0
    sin_temp = 0.0
    fitness = 0.0
    for i in range(0, NUMBER_OF_DIMENSIONS):
        temp += (data[i] + ((-1)**(i+1))*((i+1)%4))**2
        sin_temp += data[i]**(i+1)
    fitness = -math.sqrt(temp) + math.sin(sin_temp)
    return fitness

#Get arguments from cmd, they go as follows {arg1: X points, arg2: N dimensions}
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

#BEST_MOVEMENT_AMOUNT = .1/pow(8,NUMBER_OF_DIMENSIONS)
#MAX_MOVEMENT_AMOUNT = 1/pow(8,NUMBER_OF_DIMENSIONS)
#BEST_MOVEMENT_AMOUNT= (0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs)
#MAX_MOVEMENT_AMOUNT = (1.0/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs)
#print(MAX_MOVEMENT_AMOUNT)
#print(BEST_MOVEMENT_AMOUNT)
#CREATION OF THE ARRAY OF X POINTS ON N DIMENSIONS => DATASET
#Creating the "followers"
DATASET = np.full((NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS), None)

# for INDEX, VALUE in ITERABLE:
for point , i in enumerate(DATASET):
    for dim, x in enumerate(i):
        DATASET[point][dim] = random.uniform(BOUNDS[0], BOUNDS[1])
#print("--*FULL DATASET BEFORE:\n",DATASET)

#Meat of the program, Merge/Fitness Function portion of the program

ELITE_DATASET = DATASET[0]
runs = 1
while True:
    prev_fitness = fitness(ELITE_DATASET)
    #print(DATASET)
    #print((0.8/math.log(NUMBER_OF_DIMENSIONS)), " ",  (NUMBER_OF_POINTS/runs*NUMBER_OF_DIMENSIONS))

    BEST_MOVEMENT_AMOUNT = ((0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
    MAX_MOVEMENT_AMOUNT = ((1.0/math.log(NUMBER_OF_DIMENSIONS)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]

    #print("Followers Step Size: ", MAX_MOVEMENT_AMOUNT)
    #print("Best_Ones Step Size: ", BEST_MOVEMENT_AMOUNT)
    #Determine best dataset => ELITE_DATASET
    # 1 point per dimension. there can only be 1 best
    #ELITE_DATASET = DATASET[0]
    #print(runs, " ", end=" ")
    for i in DATASET:
        #print(i, fitness(i))
        if(fitness(i) > fitness(ELITE_DATASET)):
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
        if (fitness(DATASET[point]) > fitness(ELITE_DATASET)) or \
            (fitness(DATASET[point]) == fitness(ELITE_DATASET)):
            
            ELITE_DATASET = copy.deepcopy(DATASET[point])

        #if (DATASET[point].all() == ELITE_DATASET.all()):

            #Hill climbing for the "trendsetter" (elite follower)
            #for i in dimensions_to_change:
            for z in range(20):
                for i in range(NUMBER_OF_DIMENSIONS):
                    TEMP = copy.deepcopy(DATASET[point])
                    
                    DATASET[point][i] += random.uniform(0,BEST_MOVEMENT_AMOUNT)
                    
                    if not (fitness(DATASET[point]) > fitness(TEMP)):
                        for x in range(NUMBER_OF_DIMENSIONS):
                            DATASET[point][x] = TEMP[x]
                    
                        DATASET[point][i] -= random.uniform(0,BEST_MOVEMENT_AMOUNT)
                        
                        if not (fitness(DATASET[point]) > fitness(TEMP)):
                            for x in range(NUMBER_OF_DIMENSIONS):
                                DATASET[point][x] = TEMP[x]
                    #if(fitness(DATASET[point]) > fitness(ELITE_DATASET)):
                        #print("HILLCLIMBING => Old Best:", fitness(ELITE_DATASET), "New Best:", fitness(DATASET[point])) 
                    ELITE_DATASET = copy.deepcopy(DATASET[point]) #pivitol line
                    

            #print("EQUAL")
            #time.sleep(.00001)
        else:
            for i in dimensions_to_change:
                if(x[i] < DIMENSIONAL_MIDPOINT[i]):
                    DATASET[point][i] += random.uniform(0, MAX_MOVEMENT_AMOUNT)
                    #DATASET[point][i] += MAX_MOVEMENT_AMOUNT
                if(x[i] > DIMENSIONAL_MIDPOINT[i]):
                    DATASET[point][i] -= random.uniform(0, MAX_MOVEMENT_AMOUNT)
                    #DATASET[point][i] -= MAX_MOVEMENT_AMOUNT
        #print(DATASET)

    runs += 1    
    #break condition
#    if runs > 10000:
    #print("FITNESS:", fitness(ELITE_DATASET))
    #print("*" * 80)
    if(prev_fitness < fitness(ELITE_DATASET)):
        print(runs, fitness(ELITE_DATASET))
    if runs > 1000: #1000:
        break

#print("--*FULL DATASET AFTER:\n", DATASET)
#print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
#print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))
#print(fitness(ELITE_DATASET))