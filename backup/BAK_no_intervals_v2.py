from __future__ import print_function
from __future__ import division

import sys
import math
import numpy as np
import random
import time
import copy
import timeit
import os.path
import datetime
import re

#Time for total program execution (approx.)
COMP_TIME = timeit.default_timer()

NUMBER_OF_POINTS = None         #Number of points
NUMBER_OF_DIMENSIONS = None     #Number of dimensions
DATASET = None                  #Set of all points and their given positions
DIMENSIONAL_MIDPOINT = None     #Set of all midpoints for each dimension
ELITE_DATASET = None            #The elite point, AKA the point with the best fitness
BOUNDS = (-8, 8)                #The global search interval
STEP_SIZE = None                #Step size for all points
ELITE_STEP_SIZE = None          #Step size for elite point
ITERS = 1000                    #Number of iterations
ELITE_CLIMBS = 10               #Number of times to hill climb for the elite point
STAGNANT_RUN_CAP = 100          #Maximum number of allowable runs for which the fitness value does not improve
FITNESS_EVALS = None            #Number of fitness evaluations (optional)
OUTPUT_FILE = None              #Handler for output file (optional)

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
elif (len(sys.argv) == 5):
    try:
        sys.argv[1] = int(sys.argv[1])
        sys.argv[2] = int(sys.argv[2])
        sys.argv[3] = int(sys.argv[3])
    except ValueError:
        print("Invalid argument types")
        exit()
    NUMBER_OF_DIMENSIONS = sys.argv[1]
    NUMBER_OF_POINTS = sys.argv[2]
    FITNESS_EVALS = sys.argv[3]
    OUTPUT_FILE = sys.argv[4]
    if os.path.isdir(OUTPUT_FILE):
        print("File '", OUTPUT_FILE, "' is a directory")
        exit()
else:
    print("Usage: \"python", sys.argv[0], "[NUMBER OF DIMENSIONS] [NUMBER OF POINTS]\"")
    print("OR\nUsage: \"python", sys.argv[0], "[NUMBER OF DIMENSIONS] [NUMBER OF POINTS] [NUMBER OF FITNESS EVALUATIONS]\"")
    print("OR\nUsage: \"python", sys.argv[0], "[NUMBER OF DIMENSIONS] [NUMBER OF POINTS] [NUMBER OF FITNESS EVALUATIONS] [PATH TO OUTPUT FILE]\"")
    exit()


#ELITE_STEP_SIZE= 1/math.log(NUMBER_OF_DIMENSIONS)
#STEP_SIZE = 1/math.log(NUMBER_OF_DIMENSIONS)
#print(STEP_SIZE)
#print(ELITE_STEP_SIZE)


if not FITNESS_EVALS:
    FITNESS_EVALS = 1

#Output file prep
if OUTPUT_FILE:
    t_num = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as outf:
            for line in reversed(list(outf)):
                if re.search('Trial', line, re.I):
                    t_num = re.search(r'\d+',line.rstrip()).group()
                    break
    OUTPUT_FILE = open(OUTPUT_FILE, "a+")
    OUTPUT_FILE.write("\n\n<{}>\n{:^80}\nDate: {}".format("+" * 90, "<- Trial #" + str(int(t_num) + 1) + " ->",\
        str(datetime.datetime.now().strftime("%Y-%m-%d"))))
    OUTPUT_FILE.write("\nTime: {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    OUTPUT_FILE.write("\n\nFitness Evaluations: {0}\n{3:>21}{1}\n{4:>21}{2}\n".format(FITNESS_EVALS,\
        NUMBER_OF_DIMENSIONS, NUMBER_OF_POINTS, "Dimensions: ", "Points: "))


print("\n{}{}\n{:>21}{}\n{:>21}{}\n".format("Fitness Evaluations: ", FITNESS_EVALS, "Dimensions: ", NUMBER_OF_DIMENSIONS,\
    "Points: ", NUMBER_OF_POINTS))

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
    completed_runs = 0
    stagnant_runs = 0

   
    ELITE_DATASET = DATASET[0]


    start_time = timeit.default_timer()

#   while True:    
    while runs <= ITERS:
        if (NUMBER_OF_DIMENSIONS != 1):
            ELITE_STEP_SIZE = ((0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
            STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
        else:
            ELITE_STEP_SIZE = ((0.8/math.log(NUMBER_OF_DIMENSIONS+1)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
            STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS+1)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
        
        #Determine best dataset => ELITE_DATASET
        # 1 point per dimension. there can only be 1 best
        for i in DATASET:
            #print(i, fitness(i))
            if fitness(i) > fitness(ELITE_DATASET):
                #print("Old Best:", fitness(ELITE_DATASET), "New Best:", fitness(i))
                ELITE_DATASET = copy.deepcopy(i[:])

        prev_fitness = fitness(ELITE_DATASET)

        #PUT IN DESMOS
        #-\frac{\arctan\left(13x\ \right)}{2}\ +\frac{1.5}{2}
        #ELITE_STEP_SIZE = (-math.atan(NUMBER_OF_DIMENSIONS * fitness(ELITE_DATASET))/2.0) + (1.8/2)
        #print("Positive Elite Step Size:", ELITE_STEP_SIZE)
        #STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]

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

            #Elite Point!
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
                        
                        #CRITICAL LINE!
                        ELITE_DATASET = copy.deepcopy(DATASET[point])

            #Non-elite Points
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
            #print(runs, fitness(ELITE_DATASET), ELITE_STEP_SIZE)

        if ((fitness(ELITE_DATASET) - prev_fitness) <= 0.0):
            stagnant_runs += 1
            if stagnant_runs == STAGNANT_RUN_CAP:
                completed_runs = runs
                stagnant_runs = 0
                runtime = timeit.default_timer() - start_time
                runs = ITERS + 1

        #if (runs % 10) == 0:
        #    print(fitness(ELITE_DATASET), prev_fitness, fitness(ELITE_DATASET) - prev_fitness)
        #    if ((fitness(ELITE_DATASET) - prev_fitness) <= .0000001):
        #        optimum_runs = runs
        #        runs = ITERS + 1

    evals += 1

    #print("--*FULL DATASET AFTER:\n", DATASET)
    #print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
    #print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))
 
    if OUTPUT_FILE:
        OUTPUT_FILE.write("\n*-->Evaluation #{}:\n".format(evals))
        OUTPUT_FILE.write("\tElite Point Coordinates: {}\n".format(ELITE_DATASET))
        OUTPUT_FILE.write("\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}\n".format("Maximum Iterations: ", ITERS,\
            "Completed Iterations: ", completed_runs, "Stagnant Iterations: ", STAGNANT_RUN_CAP, "Actual Iterations: ",\
            completed_runs - STAGNANT_RUN_CAP))
        OUTPUT_FILE.write("\n\t{:>25}{:.5f} sec\n".format("Approximate Runtime: ", runtime))
        OUTPUT_FILE.write("\n\t{:>24} {} ***\n".format("*** FITNESS:", fitness(ELITE_DATASET)))


    print("{}".format("*-->Evaluation #" + str(evals)) + ":")
    print("\t{}".format("Elite Point Coordinates: " + str(ELITE_DATASET)))
    print("\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}".format("Maximum Iterations: ", ITERS, "Completed Iterations: ", completed_runs,\
        "Stagnant Iterations: ", STAGNANT_RUN_CAP, "Actual Iterations: ", completed_runs - STAGNANT_RUN_CAP))
    print("\n\t{:>24} {:.5f} sec\n".format("Approximate Runtime:", runtime))
    print("\t{:>25}{} ***\n".format("*** FITNESS: ",fitness(ELITE_DATASET)))
    #print("\tRuns:", optimum_runs)
    

t_runtime = timeit.default_timer() - COMP_TIME

print("\nTotal Approximate Runtime: {:.5f} sec\n".format(t_runtime))

if OUTPUT_FILE:
    OUTPUT_FILE.write("\n\nTotal Approximate Runtime: {:.5f} sec\n\n".format(t_runtime))
    OUTPUT_FILE.close()








