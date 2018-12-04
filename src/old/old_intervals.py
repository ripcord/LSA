from __future__ import print_function

import sys
import math
import numpy as np
import random
import os
import time
import copy

NUMBER_OF_POINTS = None         #Number of followers (wolves)
NUMBER_OF_DIMENSIONS = None     #Number of dimensions
DATASET = None                  #Set of all followers and their given positions
DIMENSIONAL_MIDPOINT = None     #Set of all midpoints for each dimension
ELITE_DATASET = None            #The best follower
BOUNDS = (-8, 8)                #The global search interval
STEP_SIZE = 0.05                #FIXED FOR NOW
ELITE_STEP_SIZE = 0.0001        #FIXED FOR NOW
INTERVAl_MIN = 3                #The minimum number of local search intervals
INTERVAl_MAX = (abs(BOUNDS[0]) + abs(BOUNDS[1])) // INTERVAl_MIN    #The maximum number of local search intervals
LOCAL_ELITES = {}
TABU = []
TABU_OFFSET = 0.0001
#LOCAL_ELITES = np.zeros(1, dtype=float)                             #A list of elites for each local search interval
#INTERVAL_WIDTH = (abs(BOUNDS[0]) + abs(BOUNDS[1])) // INTERVAl_MIN   #The minimum interval width...MINIMUM MUST BE DIVISOR!!!! DONT TOUCH!!

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

#Tabu function to blacklist previously
#generated points within a local interval
def tabu(follower):
    global TABU
    for i in TABU:
        #print("Tabu check:",i)
        if (follower + i) <= TABU_OFFSET:
        #    print("Tabu gap plus too low")
            return True
        elif (follower - i) <= TABU_OFFSET:
        #    print("Tabu gap minus too low")
            return True
        else:
        #    print("Continuing...")
            continue
    TABU.append(follower)
    return False

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


#NOTE: "for INDEX, VALUE in ITERABLE"


#Generating search intervals
random.seed()
INTERVALS = range(BOUNDS[0], BOUNDS[1] + 1, random.choice(range(INTERVAl_MIN, INTERVAl_MAX + 1)))
INTERVALS[-1] = BOUNDS[1]
LOCAL_INTERVALS = {}
print("--*GLOBAL SEARCH INTERVAL:\n\t[", BOUNDS[0], ",", BOUNDS[1], "]")
print("--*LOCAL SEARCH INTERVALS:")
for i, value in enumerate(INTERVALS):
    if (i == (len(INTERVALS) - 1)):
        break
    else:
        print("\tInterval:", i, "[", value, ",", INTERVALS[i+1], "]")
        LOCAL_INTERVALS[value] = INTERVALS[i + 1]

#CREATION OF THE ARRAY OF X POINTS ON N DIMENSIONS => DATASET
#Creating the "followers"
DATASET = np.full((NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS), None)

#Iterating through search intervals
index = 0
for LOW in sorted(LOCAL_INTERVALS.keys()):

    TABU = []

    for point , i in enumerate(DATASET):
        for dim, x in enumerate(i):
            DATASET[point][dim] = random.uniform(LOW, LOCAL_INTERVALS[LOW])
            while (tabu(DATASET[point][dim])):
                DATASET[point][dim] = random.uniform(LOW, LOCAL_INTERVALS[LOW])
                print("New Data Point:",DATASET[point][dim])
                print(TABU)
    print("--*FULL DATASET BEFORE:\n",DATASET)
    print("TABU: ", TABU)
    exit()

    runs = 0
    #Meat of the program, Merge/Fitness Function portion of the program

    ELITE_DATASET = DATASET[0]

    #while runs < 100000:
    #while runs < 10000:
    while runs < 1000:
        #Determine best dataset => ELITE_DATASET
        # 1 point per dimension. there can only be 1 best
#!ME    #    ELITE_DATASET = DATASET[0]
        for i in DATASET:
            #print(i, fitness(i))
            if(fitness(i) > fitness(ELITE_DATASET)):
                ELITE_DATASET = copy.deepcopy(i[:])

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
            #if (DATASET[point].all() == ELITE_DATASET.all()):
            
            if (fitness(DATASET[point]) >= fitness(ELITE_DATASET)):
                #Hill climbing for the "trendsetter" (elite follower)
            #RANDOM NUMBER OF DIMENSIONS
                #for i in dimensions_to_change:
            #ALL DIMENSIONS
                for i in range(NUMBER_OF_DIMENSIONS):
#                    print(DATASET[point])
                    TEMP = copy.deepcopy(DATASET[point])
#                    print(TEMP)
#                    print("About to increment elite step size...")
                    
                    DATASET[point][i] += ELITE_STEP_SIZE
#                    print("Incremented our step size...")
#                    print(DATASET[point][i])
                    
                    if not (fitness(DATASET[point]) > fitness(TEMP)):
#                        print("Decrementing our elite step size...")
                        DATASET[point][i] = TEMP[i]
                       
#                        print(DATASET[point][i])
#                        print(TEMP[i])
                        DATASET[point][i] -= ELITE_STEP_SIZE
                        
#                        print("Decremented our elite step size...")
#                        print(DATASET[point][i])

                        if not (fitness(DATASET[point]) > fitness(TEMP)):
#                           print("Gonna reset our elite step size...")
                            DATASET[point][i] = TEMP[i]
#                           print(DATASET[point][i])
#                           print(TEMP[i])


                ELITE_DATASET = copy.deepcopy(DATASET[point])
#               print("Our final elite: ",ELITE_DATASET)

                #print("EQUAL")
                #time.sleep(.00001)
            else:
                for i in dimensions_to_change:
                    if(x[i] < DIMENSIONAL_MIDPOINT[i]):
                        #DATASET[point][i] += random.uniform(0, STEP_SIZE)
                        DATASET[point][i] += STEP_SIZE
                    if(x[i] > DIMENSIONAL_MIDPOINT[i]):
                        #DATASET[point][i] -= random.uniform(0, STEP_SIZE)
                        DATASET[point][i] -= STEP_SIZE
            #print(DATASET)

        runs += 1


    LOCAL_ELITES[index] = ELITE_DATASET
    index += 1

    #LOCAL_ELITES = np.append(LOCAL_ELITES, ELITE_DATASET)


for i in LOCAL_ELITES:
    #print(LOCAL_ELITES)
    #print(ELITE_DATASET)
    if fitness(LOCAL_ELITES[i]) > fitness(ELITE_DATASET):
        ELITE_DATASET = copy.deepcopy(LOCAL_ELITES[i])

print("--*FULL DATASET AFTER:\n", DATASET)
print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))

