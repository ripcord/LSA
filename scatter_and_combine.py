from __future__ import print_function

import sys
import math
import numpy as np
import random
import time


NUMBER_OF_POINTS = None
NUMBER_OF_DIMENSIONS = None
DATASET = None
DIMENSIONAL_MIDPOINT = None
ELITE_DATASET = None
BOUNDS = (-8, 8)
MAX_MOVEMENT_AMOUNT = 0.1 #FIXED FOR NOW

#here is defined the fitness function
#FOR THE FUTURE, THIS MAY BE BETTER TO PUT OUTSIDE OF THE PROGRAM, IN A SEPREATE
#CLASS TO BE USED BY OTHER PROGRAMS
#this is the elvis needs boats problem, you must send it a whole set of data
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
DATASET = np.full((NUMBER_OF_DIMENSIONS, NUMBER_OF_POINTS), None)
#print(DATASET.shape)
#SAMPLE: [None None None None None None] [None None None None None None] 
#        ^ each row is a dimension, each column is a point

#Now I will use BOUNDS as the search space and pick random values within that
#space for the initial values
for dim , i in enumerate(DATASET):
    for point, x in enumerate(i):
        DATASET[dim][point] = random.uniform(BOUNDS[0], BOUNDS[1])
print("--*FULL DATASET BEFORE:\n",DATASET)
print("Minimum: ", np.min(DATASET[0]), "Maximum: ", np.max(DATASET[0]))

runs = 0
#Meat of the program, Merge/Fitness Function portion of the program
while True:
    #Determine best dataset => ELITE_DATASET
    # 1 point per dimension. there can only be 1 best
    ELITE_DATASET = DATASET.T[0]
    for i in DATASET.T:
        #print(i, fitness(i))
        if(fitness(i) > fitness(ELITE_DATASET)):
            ELITE_DATASET = i[:]

    #Find the midpoint of all the points
    DIMENSIONAL_MIDPOINT = np.full((NUMBER_OF_DIMENSIONS, 1), None)
    for index, i in enumerate(DATASET):
        DIMENSIONAL_MIDPOINT[index] = np.mean(i.T)
    DIMENSIONAL_MIDPOINT = DIMENSIONAL_MIDPOINT.T[0]
    #print("THE MIDPOINT", DIMENSIONAL_MIDPOINT)

    

    #random dimensions to change
    dimensions_to_change = random.sample(range(0, NUMBER_OF_DIMENSIONS), random.randint(0, NUMBER_OF_DIMENSIONS))
    #print(dimensions_to_change)
    
    #move in those specific random dimensions toward middle
    for i in dimensions_to_change:
        #print(i, ":DATASET:", DATASET[i].T, "\nELITE_DATASET", ELITE_DATASET)
        if(DATASET[i].T.all() == ELITE_DATASET.all()):
            #send the best data to THE SEARCH ALGORITHM!!!!
            my_little_step_size = .001
            for point, x in enumerate(DATASET[i]):
                x[point] += random.uniform(0, my_little_step_size)

            time.sleep(.0001)
        else:
            #make the rest of the points move toward the center
            for point, x in enumerate(DATASET[i]):
                if(x < DIMENSIONAL_MIDPOINT[i]):
                    DATASET[i][point] += random.uniform(0, MAX_MOVEMENT_AMOUNT)
                if(x > DIMENSIONAL_MIDPOINT[i]):
                    DATASET[i][point] -= random.uniform(0, MAX_MOVEMENT_AMOUNT)
        #print("THE DATASET", DATASET)
    runs += 1    
    #break condition
    if runs > 50:
        break



print("--*FULL DATASET AFTER:\n", DATASET)
print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
print("--*BEST DATA:", ELITE_DATASET, fitness(ELITE_DATASET))

