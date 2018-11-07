import random
import math
import numpy as np
import copy

PACK_SIZE = 3 #Wolves per pack
DIMENSIONS = 2 #N dimensions
STEP = 0.25 #Step size toward wolf with high fitness
SEARCH_RANGE = 3.0 #search in range -x -- x

current_pack = np.zeros(shape=(PACK_SIZE, DIMENSIONS))
experimental_pack = np.zeros(shape=(PACK_SIZE, DIMENSIONS))
fittest_pack = [[None] * DIMENSIONS] * PACK_SIZE
high_fitness = 0.0
#top_wolf = None

def individual_fitness(wolf):
    temp = 0.0
    sin_temp = 0.0
    fitness = 0.0
    for i in range(0, DIMENSIONS):
        temp += (wolf[i] + ((-1)**(i+1))*((i+1)%4))**2
        sin_temp += wolf[i]**(i+1)
    fitness = -math.sqrt(temp) + math.sin(sin_temp)
    return fitness

#init fill/ random restart
#random distribution of PACK_SIZE wolves
def init():
    global current_pack
    global high_fitness
    for wolf in range(0, PACK_SIZE):
        for dim in range(0, DIMENSIONS):
            current_pack[wolf][dim] = random.uniform(-SEARCH_RANGE, SEARCH_RANGE)
    high_fitness = individual_fitness(current_pack[0])
    for wolf in current_pack:
        if individual_fitness(wolf) > high_fitness:
            high_fitness = individual_fitness(wolf)
            #top_wolf = wolf;
    return

def merge():
    global experimental_pack
    #global top_wolf
    pack_center = np.zeros(shape=(DIMENSIONS))
    best_wolf = experimental_pack[0]
    
    #find current global maxima
    for wolf in range(0, PACK_SIZE):
        if(individual_fitness(experimental_pack[wolf]) > individual_fitness(best_wolf)):
            best_wolf = experimental_pack[wolf]
            #if(top_wolf is None):
                #top_wolf = best_wolf
            #elif(individual_fitness(best_wolf) > individual_fitness(top_wolf)):
                #top_wolf = best_wolf
    
    #find center of pack
    for dim in range(0, DIMENSIONS):
        pack_center[dim] = np.average(experimental_pack.T[dim])
    
    #move all wolves toward center of pack except the one with the highest fitness
    for wolf in experimental_pack:
        if((wolf == best_wolf).all()):
            #he has food, so stay there
            stay()
        else:
            for dim in range(0, DIMENSIONS):
                if(wolf[dim] < pack_center[dim]):
                    wolf[dim] += random.uniform(0, STEP)
                elif(wolf[dim] > pack_center[dim]):
                    wolf[dim] -= random.uniform(0, STEP)
    return

def stay():
    #stay boy stay
    return

def main():
    global fittest_pack
    global experimental_pack
    global current_pack

    init()
    #print(current_pack)
    
    fittest_pack = copy.deepcopy(current_pack)
    experimental_pack = copy.deepcopy(current_pack)
    
    
    init()
    experimental_pack = copy.deepcopy(current_pack) 
    print(experimental_pack)
    for i in range(500):
        merge()
    print(experimental_pack)
    #print("Top Wolf", top_wolf)
    #print(individual_fitness(top_wolf))
    return



main()