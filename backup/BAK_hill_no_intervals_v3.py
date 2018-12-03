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
BOUNDS = None                   #The global search interval
STEP_SIZE = None                #Step size for all non-elite points
ELITE_STEP_SIZE = None          #Step size for elite point
ITERS = 1000                    #Number of iterations
ELITE_CLIMBS = 10               #Number of times to hill climb for the elite point
STAGNANT_RUN_CAP = 100          #Maximum number of allowable runs for which the fitness value does not improve
RESULTS = {}                    #Set of evaluation result information (actual iterations, fitness, elite coordinates, evaluation number)
FITNESS_EVALS = None            #Number of fitness evaluations (optional)
OUTPUT_FILE = None              #Handler for output file (optional)
FITNESS_MIN = False             #Switch to enable finding global minimum rather than maximum


#Fitness function "Elvis Needs Boats"
#2D optimum = ~0.41
def elvis_needs_boats(data):
    global NUMBER_OF_DIMENSIONS
    temp = 0.0
    sin_temp = 0.0
    fitness = 0.0
    for i in range(0, NUMBER_OF_DIMENSIONS):
        temp += (data[i] + ((-1)**(i+1))*((i+1)%4))**2
        sin_temp += data[i]**(i+1)
    fitness = -math.sqrt(temp) + math.sin(sin_temp)
    return fitness

#Fitness function "Rastrigin"
#Global minimum at "f(0,...,0) = 0"
def rastrigin(data):
    global NUMBER_OF_DIMENSIONS
    fitness = 0.0
    for i in range(0, NUMBER_OF_DIMENSIONS):
        fitness += (data[i]**2) - 10 * math.cos(2 * math.pi * data[i])
    fitness += (10 * NUMBER_OF_DIMENSIONS)
    return fitness

def eggholder(data):
    global NUMBER_OF_DIMENSIONS
    fitness = 0.0
    if NUMBER_OF_DIMENSIONS > 2:
        print("Error: Eggholder function only accepts 2 dimensions")
        exit()
    else:
        fitness = -(data[1] + 47) * np.sin(np.sqrt(abs(data[1] + (data[0] /2) + 47))) + (-data[0] * np.sin(np.sqrt(abs(data[0] - (data[1] - 47)))))
    return fitness

def fitness(data):
    return elvis_needs_boats(data)
    #return rastrigin(data)
    #return eggholder(data)

#Compares fitness of two values "a" and "b"
def comp_fitness(a, b, equals=None):
    global FITNESS_MIN
    if FITNESS_MIN:
        if equals:
            return (fitness(a) <= fitness(b))
        else:
            return (fitness(a) < fitness(b))
    else:
        if equals:
            return (fitness(a) >= fitness(b))
        else:
            return (fitness(a) > fitness(b))

#Returns a step size, variable for elite and non-elite points
def get_step(runs, elite=False, nd_offset=None):
    global NUMBER_OF_DIMENSIONS
    global NUMBER_OF_POINTS
    global BOUNDS
    if elite:
        if nd_offset:
            return ((0.8/math.log(NUMBER_OF_DIMENSIONS + nd_offset)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
        else:
            return ((0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
    else:
        if nd_offset:
            return ((1.0/math.log(NUMBER_OF_DIMENSIONS + nd_offset)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
        else:
            return ((1.0/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]

#Sets global search interval (optimization function-dependent)
def get_bounds():
    if fitness((2, 2)) == elvis_needs_boats((2, 2)):
        return (-8, 8)
    elif fitness((2, 2)) == rastrigin((2, 2)):
        return (-5.12, 5.12)
    elif fitness((2, 2)) == eggholder((2, 2)):
        return (-512, 512)
    else:
        return (-8, 8)

#Output to a file and/or STDOUT
def output(evaluations, comp_runs, stag_runs, l_runtime, out_file=None):
    global ELITE_DATASET
    global ITERS
    global STAGNANT_RUN_CAP
    if out_file:
        out_file.write("\n*-->Evaluation #{}:\n".format(evaluations))
        out_file.write("\tElite Point Coordinates: {}\n".format(ELITE_DATASET))
        out_file.write("\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}\n".format("Maximum Iterations: ", ITERS,\
            "Completed Iterations: ", comp_runs, "Stagnant Iterations: ", stag_runs, "Actual Iterations: ",\
            comp_runs - stag_runs))
        out_file.write("\n\t{:>25}{:.5f} sec\n".format("Approximate Runtime: ", l_runtime))
        out_file.write("\n\t{:>24} {} ***\n".format("*** FITNESS:", fitness(ELITE_DATASET)))
    else:
        print("*-->Evaluation #{}:".format(evaluations))
        print("\tElite Point Coordinates: {}".format(ELITE_DATASET))
        print("\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}\n\t{:>25}{}".format("Maximum Iterations: ", ITERS, "Completed Iterations: ", comp_runs,\
            "Stagnant Iterations: ", stag_runs, "Actual Iterations: ", comp_runs - stag_runs))
        print("\n\t{:>24} {:.5f} sec\n".format("Approximate Runtime:", l_runtime))
        print("\t{:>25}{} ***\n".format("*** FITNESS: ",fitness(ELITE_DATASET)))

    return

#Writes logging headers to an output file
def file_prep():
    global OUTPUT_FILE
    global FITNESS_EVALS
    global NUMBER_OF_DIMENSIONS
    global NUMBER_OF_POINTS
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
    OUTPUT_FILE.write("\nTime: {}".format(datetime.datetime.now().strftime("%H:%M.%S")))
    OUTPUT_FILE.write("\n\nFitness Evaluations: {0}\n{3:>21}{1}\n{4:>21}{2}\n{5:>21}{6}\n".format(FITNESS_EVALS,\
        NUMBER_OF_DIMENSIONS, NUMBER_OF_POINTS, "Dimensions: ", "Points: ", "Optimum: ", "Minimum" if FITNESS_MIN else "Maximum"))
    return

#Writes the best fitness to STDOUT and/or an output file
def print_best(data, out_file=None):
    global FITNESS_MIN
    i = 1
    temp = data[i][0]
    for k in sorted(data.keys()):
        if FITNESS_MIN:
            if data[k][0] < temp:
                temp = data[k][0]
                i = k
        else:
            if data[k][0] > temp:
                temp = data[k][0]
                i = k
    print("\n\t{0:>19}\n\t{1:>19}\n\t{0:>19}".format("-" * 18,"| Global Minimum |" if FITNESS_MIN else "| Global Maximum |"))
    print("{:>18} {}\nActual Iterations: {}\nElite Coordinates: {}\n{:>18} {}".format("Best Evaluation:", i, data[i][2],\
            data[i][1], "Fitness:", temp))
    if out_file:
        out_file.write("\n\n\t{0:>19}\n\t{1:>19}\n\t{0:>19}".format("-" * 18,"| Global Minimum |" if FITNESS_MIN else "| Global Maximum |"))
        out_file.write("\n{:>18} {}\nActual Iterations: {}\nElite Coordinates: {}\n{:>18} {}".format("Best Evaluation:",\
            i, data[i][2], data[i][1], "Fitness:", temp))
    return

def print_usage():
    print("\nNote: The argument 'OPTIMUM TYPE' accepts values of '+' or '-'. Value of '+' will produce global maxima, while \
'-' will produce global minima.")
    print("\nUse cases:\n\t\"python", sys.argv[0], "[OPTIMUM TYPE (max = '+', min = '-')] [DIMENSIONS] [POINTS]\"")
    print("\n\t\"python", sys.argv[0], "[OPTIMUM TYPE (max = '+', min = '-')] [DIMENSIONS] [POINTS] [FITNESS EVALUATIONS]\"")
    print("\n\t\"python", sys.argv[0], "[OPTIMUM TYPE (max = '+', min = '-')] [DIMENSIONS] [POINTS] [FITNESS EVALUATIONS] \
[PATH TO OUTPUT FILE]\"")


#Get arguments from cmd, they go as follows {arg1: X points, arg2: N dimensions}
#X points on N dimensions
if ("?" in sys.argv) or (len(sys.argv) == 0):
    print_usage()
    exit()
if (len(sys.argv) == 4):
    try:
        sys.argv[2] = int(sys.argv[2])
        sys.argv[3] = int(sys.argv[3])
    except ValueError:
        print("Number of dimensions and points must be of type integer")
        exit()
    NUMBER_OF_DIMENSIONS = sys.argv[2]
    NUMBER_OF_POINTS = sys.argv[3]
    if (NUMBER_OF_DIMENSIONS <= 0) or (NUMBER_OF_POINTS <= 0):
        print("Number of dimensions and points must be greater than 0")
        exit()
    if sys.argv[1] == "+":
        FITNESS_MIN = False
    elif sys.argv[1] == "-":
        FITNESS_MIN = True
    else:
        print("Invalid optimum type. Acceptable values: '+' (maximum), '-' (minimum)")
        exit()
elif (len(sys.argv) == 5):
    try:
        sys.argv[2] = int(sys.argv[2])
        sys.argv[3] = int(sys.argv[3])
        sys.argv[4] = int(sys.argv[4])
    except ValueError:
        print("Number of dimensions, points and fitness evaluations must be of type integer")
        exit()
    NUMBER_OF_DIMENSIONS = sys.argv[2]
    NUMBER_OF_POINTS = sys.argv[3]
    FITNESS_EVALS = sys.argv[4]
    if (NUMBER_OF_DIMENSIONS <= 0) or (NUMBER_OF_POINTS <= 0) or (FITNESS_EVALS <= 0):
        print("Number of dimensions, points and fitness evaluations must be greater than 0")
        exit()
    if sys.argv[1] == "+":
        FITNESS_MIN = False
    elif sys.argv[1] == "-":
        FITNESS_MIN = True
    else:
        print("Invalid optimum type. Acceptable values: '+' (maximum), '-' (minimum)")
        exit()
elif (len(sys.argv) == 6):
    try:
        sys.argv[2] = int(sys.argv[2])
        sys.argv[3] = int(sys.argv[3])
        sys.argv[4] = int(sys.argv[4])
    except ValueError:
        print("Number of dimensions, points and fitness evaluations must be of type integer")
        exit()
    NUMBER_OF_DIMENSIONS = sys.argv[2]
    NUMBER_OF_POINTS = sys.argv[3]
    FITNESS_EVALS = sys.argv[4]
    OUTPUT_FILE = sys.argv[5]
    if os.path.isdir(OUTPUT_FILE):
        print("File '", os.path.abspath(OUTPUT_FILE), "' is a directory")
        exit()
    if (NUMBER_OF_DIMENSIONS <= 0) or (NUMBER_OF_POINTS <= 0) or (FITNESS_EVALS <= 0):
        print("Number of dimensions, points and fitness evaluations must be greater than 0")
        exit()
    if sys.argv[1] == "+":
        FITNESS_MIN = False
    elif sys.argv[1] == "-":
        FITNESS_MIN = True
    else:
        print("Invalid optimum type. Acceptable values: '+' (maximum), '-' (minimum)")
        exit()
else:
    print_usage()
    exit()


BOUNDS = get_bounds()

#ELITE_STEP_SIZE= 1/math.log(NUMBER_OF_DIMENSIONS)
#STEP_SIZE = 1/math.log(NUMBER_OF_DIMENSIONS)
#print(STEP_SIZE)
#print(ELITE_STEP_SIZE)

if not FITNESS_EVALS:
    FITNESS_EVALS = 1

#Output file prep
if OUTPUT_FILE:
    file_prep()


print("\nFitness Evaluations: {}\n{:>21}{}\n{:>21}{}\n{:>21}{}\n".format(FITNESS_EVALS, "Dimensions: ", NUMBER_OF_DIMENSIONS,\
    "Points: ", NUMBER_OF_POINTS, "Optimum: ","Minimum" if FITNESS_MIN else "Maximum"))

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
    runtime = 0.0

   
    ELITE_DATASET = DATASET[0]


    start_time = timeit.default_timer()

#   while True:    
    while runs <= ITERS:
        if (NUMBER_OF_DIMENSIONS != 1):
            #ELITE_STEP_SIZE = ((0.8/math.log(NUMBER_OF_DIMENSIONS)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
            #STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
            ELITE_STEP_SIZE = get_step(runs, True)
            STEP_SIZE = get_step(runs)
        else:
            #ELITE_STEP_SIZE = ((0.8/math.log(NUMBER_OF_DIMENSIONS+1)) * (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
            #STEP_SIZE = ((1.0/math.log(NUMBER_OF_DIMENSIONS+1)) *  (NUMBER_OF_POINTS/runs*4)) % BOUNDS[1]
            ELITE_STEP_SIZE = get_step(runs, True, 1)
            STEP_SIZE = get_step(runs, False, 1)
        
        #Determine best dataset => ELITE_DATASET
        # 1 point per dimension. there can only be 1 best
        for i in DATASET:
            #print(i, fitness(i))

####
    #        if fitness(i) > fitness(ELITE_DATASET):
            if comp_fitness(i, ELITE_DATASET):
####
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

###
            #Elite Point!
    #        if fitness(DATASET[point]) >= fitness(ELITE_DATASET):
            if comp_fitness(DATASET[point], ELITE_DATASET, "="):
 ###               
                ELITE_DATASET = copy.deepcopy(DATASET[point])

                #Hill climbing for the elite point
                for climb in range(ELITE_CLIMBS):
                    for i in range(NUMBER_OF_DIMENSIONS):
                        TEMP = copy.deepcopy(DATASET[point])
                        
                        DATASET[point][i] += random.uniform(0,ELITE_STEP_SIZE)
                        
###
   #                     if not (fitness(DATASET[point]) > fitness(TEMP)):
                        if not comp_fitness(DATASET[point], TEMP):
###
                            for x in range(NUMBER_OF_DIMENSIONS):
                                DATASET[point][x] = TEMP[x]
                        
                            DATASET[point][i] -= random.uniform(0,ELITE_STEP_SIZE)
                            
###
   #                         if not (fitness(DATASET[point]) > fitness(TEMP)):
                            if not comp_fitness(DATASET[point], TEMP):
###
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

###
   #     if ((fitness(ELITE_DATASET) - prev_fitness) <= 0.0):
        if FITNESS_MIN:
            if ((fitness(ELITE_DATASET) - prev_fitness) >= 0.0):
                stagnant_runs += 1
        else:
            if ((fitness(ELITE_DATASET) - prev_fitness) <= 0.0):
                stagnant_runs += 1
        if stagnant_runs == STAGNANT_RUN_CAP:
            completed_runs = runs
            runtime = timeit.default_timer() - start_time
            runs = ITERS + 1
###

        #if (runs % 10) == 0:
        #    print(fitness(ELITE_DATASET), prev_fitness, fitness(ELITE_DATASET) - prev_fitness)
        #    if ((fitness(ELITE_DATASET) - prev_fitness) <= .0000001):
        #        optimum_runs = runs
        #        runs = ITERS + 1

    if stagnant_runs == 0:
        completed_runs = runs - 1
        runtime = timeit.default_timer() - start_time

    evals += 1

    #This needs to be after "evals += 1" !!!
    RESULTS[evals] = (fitness(ELITE_DATASET), ELITE_DATASET, completed_runs - stagnant_runs)

    #print("--*FULL DATASET AFTER:\n", DATASET)
    #print("--*DIMENSION MIDPOINT:\n", DIMENSIONAL_MIDPOINT)
    #print("--*BEST DATA:", ELITE_DATASET, "| FITNESS:", fitness(ELITE_DATASET))
 
    if OUTPUT_FILE:
        output(evals, completed_runs, stagnant_runs, runtime, OUTPUT_FILE)

    output(evals, completed_runs, stagnant_runs, runtime)
    #print("\tRuns:", optimum_runs)

if OUTPUT_FILE:
    print_best(RESULTS, OUTPUT_FILE)
else:
    print_best(RESULTS)

t_runtime = timeit.default_timer() - COMP_TIME

print("\nTotal Approximate Runtime: {:.5f} sec\n".format(t_runtime))

if OUTPUT_FILE:
    OUTPUT_FILE.write("\n\nTotal Approximate Runtime: {:.5f} sec\n\n".format(t_runtime))
    OUTPUT_FILE.close()





