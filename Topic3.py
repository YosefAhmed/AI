from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import sys
r = random.Random()
r.seed("AI")


import math


# region SearchAlgorithms
class Stack:

    def __init__(self):
        self.stack = []

    def push(self, value):
        if value not in self.stack:
            self.stack.append(value)
            return True
        else:
            return False

    def exists(self, value):
        if value not in self.stack:
            return True
        else:
            return False

    def pop(self):
        if len(self.stack) <= 0:
            return ("The Stack == empty")
        else:
            return self.stack.pop()

    def top(self):
        return self.stack[0]


class Node:
    id =-1
    up =-1
    down =-1
    left = -1
    right = -1
    previousNode = -1
    edgeCost = -1
    gOfN = -1 # total edge cost
    hOfN = -1 # heuristic value
    heuristicFn = -1

    def __init__(self, value, id, up, down, left, right, prevNode, edgeCost, gOfN, hOfN,fOfN):
        self.value = value
        self.id = id
        self.up=up
        self.down=down
        self.left=left
        self.right=right
        self.previousNode=prevNode
        self.edgeCost=edgeCost
        self.gOfN=gOfN
        self.hOfN=hOfN
        self.heuristicFn=fOfN



class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    edgeCosts = None #2D array represents the edge costs
    nodes = [] #array of nodes
    startNode = None #[value, X, Y]
    goalNode = None #[value, X, Y]
    open = []
    close = []
   #------------Main class properties and functions  ---------------------
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    totalCost = None

    def __init__(self, mazeStr,edgeCost = None):
        column = mazeStr.split(' ')
        for i in range(len(column)):
            self.nodes.append([])
            row = column[i].split(',')
        # -- finding startNode and endNode --
            if self.startNode is None:
                self.GetStartAndGoal(mazeStr.index('S'),mazeStr.index('E'), len(column), len(column[0]), mazeStr)
        # ------------------------------------------------------------------------
            for j in range(len(row)):

                self.nodes[i].append(Node(row[j], i*len(row)+j, None, None, None, None,
                                 -1, -1, 0, self.Heuristic(j,i), sys.maxsize ))

                #edge cost
                self.nodes[i][j].edgeCost = edgeCost[i*len(row)+j]
                #left
                if j is not 0:
                   self.nodes[i][j].left = [i, j-1]
                #right
                if j is not len(row)-1:
                    self.nodes[i][j].right = [i, j+1]
                #up
                if i is not 0:
                    self.nodes[i][j].up = [i-1, j]

                    #down
                    self.nodes[i-1][j].down = [i,j]

    def AstarManhattanHeuristic(self):
        #seting start node
        self.nodes[self.startNode[1]][self.startNode[2]].heuristicFn = \
            self.nodes[self.startNode[1]][self.startNode[2]].hOfN + self.nodes[self.startNode[1]][self.startNode[2]].gOfN

        #append start node to Open list
        self.open.append(self.nodes[self.startNode[1]][self.startNode[2]])

        prevNode = None
        while len(self.open) != 0:

        # -- sorting the Open list ordered by heuristicFn --
            self.open.sort(key= lambda node: node.heuristicFn)

        # -- pop the least order node to get its neighbours and append it to Close list --
            currentNode = self.open.pop(0)
            # ==handel the case of trying a wrong path (if the selected path was blocked)==
            if prevNode is not None and currentNode.previousNode is not prevNode.id:
                while currentNode.previousNode is not prevNode.id:
                    prevNode = self.close.pop()
            #==============================================================================

        # -- update attributes of neighbours of the current node ( update parent, gOfN, FoFN), and append them to Open list --
            if currentNode.up != None:
                up = self.nodes[currentNode.up[0]][currentNode.up[1]]
                if(currentNode.heuristicFn < up.heuristicFn):
                    up.previousNode = currentNode.id
                    up.gOfN = currentNode.gOfN +  up.edgeCost
                    up.heuristicFn = up.gOfN + up.hOfN
                    self.open.append(up)
            if currentNode.down != None:
                down = self.nodes[currentNode.down[0]][currentNode.down[1]]
                if (currentNode.heuristicFn < down.heuristicFn):
                    down.previousNode = currentNode.id
                    down.gOfN =  down.edgeCost + currentNode.gOfN
                    down.heuristicFn = down.gOfN + down.hOfN
                    self.open.append(down)
            if currentNode.left != None:
                left = self.nodes[currentNode.left[0]][currentNode.left[1]]
                if (currentNode.heuristicFn < left.heuristicFn):
                    left.previousNode = currentNode.id
                    left.gOfN = currentNode.gOfN +  left.edgeCost
                    left.heuristicFn = left.gOfN + left.hOfN
                    self.open.append(left)
            if currentNode.right != None:
                right = self.nodes[currentNode.right[0]][currentNode.right[1]]
                if (currentNode.heuristicFn < right.heuristicFn):
                    right.previousNode = currentNode.id
                    right.gOfN = currentNode.gOfN +  right.edgeCost
                    right.heuristicFn = right.gOfN + right.hOfN
                    self.open.append(right)
        # ---------------------------------------------------------------------------------------------------------------------------------

        # -- append the current node to Close list and append its id to the fullPath --
            self.close.append(currentNode)
            self.fullPath.append(currentNode.id)

        # -- if the goal is reached --
            if currentNode.value == self.goalNode[0]:
                for obj in self.close:
                    self.path.append(obj.id)
                self.totalCost = currentNode.gOfN
                break

            prevNode = currentNode
        # print (self.nodes[0].hOfN)
        self.PrintMaze(self.nodes)
        return self.fullPath, self.path, self.totalCost

    # ------------Main class properties and functions  ---------------------

    def GetStartAndGoal(self, startIndex, goalIndex, columnSize, strRowSize, str):

        startTmpX = startTmpY = goalTmpX = goalTmpY = startNodeX = startNodeY = goalNodeX = goalNodeY = 0

        startNodeY = int(startIndex / (strRowSize+1))
        goalNodeY = int(goalIndex / (strRowSize+1))

        if startNodeY == 0:
            for i in range(strRowSize-1):
                if str[i] == 'S':
                    startNodeX = startTmpX
                    break
                startTmpX+=1
        else:
            for i in range(startNodeY*(strRowSize+1), startNodeY*strRowSize + (strRowSize+1)):
                k = str[i]
                if str[i] == 'S':
                    startNodeX = startTmpX
                    break
                if str[i] != ',': startTmpX+=1

        if goalNodeY == 0:
            for i in range(strRowSize-1):
                if str[i] == 'E':
                    goalNodeX = goalTmpX
                    break
                goalTmpX+=1
        else:
            for i in range(goalNodeY*(strRowSize+1), goalNodeY*strRowSize + (strRowSize)):
                k = str[i]
                if str[i] == 'E':
                    goalNodeX = goalTmpX
                    break
                if str[i] != ',': goalTmpX+=1

        self.startNode = ['S', startNodeY, startNodeX]
        self.goalNode = ['E', goalNodeY, goalNodeX]

        # indxInColumn = [startIndex - (columnSize - 1), goalIndex - (columnSize - 1)]
        # for e in range(1):
        #     if indxInColumn[e] < 0:
        #         indxInColumn[e] = 0
        # startIndex = indxInColumn[0]
        # goalIndex = indxInColumn[1]
        #
        # i = 0
        # # startNode
        # startNodeY =int( startIndex / rowSize)
        #
        # i += startNodeY * rowSize
        # if startNodeY == 0:
        #     for c in range(i,rowSize-1):
        #         if str[c] == 'S':
        #             startNodeX = c
        #             break
        # elif startNodeY == columnSize-1:
        #     c=0
        #     for g in range(i,i+rowSize-1):
        #         tmp = str[g]
        #         if str[g] == 'S':
        #             startNodeX = c - (int(rowSize - (rowSize / 2)))
        #             break
        #         c+=1
        #
        # else:
        #     if str[i] == 'S':
        #         startNodeX = i
        #     elif str[i+(rowSize)] == 'S':
        #         startNodeX = int(rowSize - (rowSize/2))
        # # startNodeX = startIndex - i
        # # startNodeX = startIndex % rowSize - ((startIndex % rowSize) / 2)
        #
        # # goalNode
        # goalNodeX = goalIndex % rowSize - ((goalIndex % rowSize) / 2)
        # goalNodeY = goalIndex / rowSize


    def Heuristic(self, x, y):
        return (abs(x-self.goalNode[2])+abs(y-self.goalNode[1]))

    def PrintMaze(self, maze):
        # print maze with edge costs
        for y in range(len(maze)):
            for x in range(len(maze[0])):
                print("(", maze[y][x].edgeCost, ")", end = " ")

            print()

            for x in range(len(maze[0])):
                if maze[y][x].value == 'S'or maze[y][x].value == 'E':
                    print (" ", maze[y][x].value, " ",end = " ")
                else:
                    print(" ", maze[y][x].id, " ", end=" ")

            print()


# endregion

# region KNN
class KNN_Algorithm:
    def __init__(self, K):
        self.K = K

    def euclidean_distance(self, p1, p2):
        pass

    def KNN(self, X_train, X_test, Y_train, Y_test):
        pass

# endregion KNN


# region GeneticAlgorithm
class GeneticAlgorithm:
    Cities = [1, 2, 3, 4, 5, 6]
    DNA_SIZE = len(Cities)
    POP_SIZE = 20
    GENERATIONS = 5000

    """
    - Chooses a random element from items, where items is a list of tuples in
       the form (item, weight).
    - weight determines the probability of choosing its respective item. 
     """

    def weighted_choice(self, items):
        weight_total = sum((item[1] for item in items))
        n = r.uniform(0, weight_total)
        for item, weight in items:
            if n < weight:
                return item
            n = n - weight
        return item

    """ 
      Return a random character between ASCII 32 and 126 (i.e. spaces, symbols, 
       letters, and digits). All characters returned will be nicely printable. 
    """

    def random_char():
        return chr(int(r.randrange(32, 126, 1)))

    """ 
       Return a list of POP_SIZE individuals, each randomly generated via iterating 
       DNA_SIZE times to generate a string of random characters with random_char(). 
    """

    def random_population(self):
        pop = []
        for i in range(1, 21):
            x = r.sample(self.Cities, len(self.Cities))
            if x not in pop:
                pop.append(x)
        return pop

    """ 
      For each gene in the DNA, this function calculates the difference between 
      it and the character in the same position in the OPTIMAL string. These values 
      are summed and then returned. 
    """

    def cost(self, city1, city2):
        if (city1 == 1 and city2 == 2) or (city1 == 2 and city2 == 1):
            return 10
        elif (city1 == 1 and city2 == 3) or (city1 == 3 and city2 == 1):
            return 20
        elif (city1 == 1 and city2 == 4) or (city1 == 4 and city2 == 1):
            return 23
        elif (city1 == 1 and city2 == 5) or (city1 == 5 and city2 == 1):
            return 53
        elif (city1 == 1 and city2 == 6) or (city1 == 6 and city2 == 1):
            return 12
        elif (city1 == 2 and city2 == 3) or (city1 == 3 and city2 == 2):
            return 4
        elif (city1 == 2 and city2 == 4) or (city1 == 4 and city2 == 2):
            return 15
        elif (city1 == 2 and city2 == 5) or (city1 == 5 and city2 == 2):
            return 32
        elif (city1 == 2 and city2 == 6) or (city1 == 6 and city2 == 2):
            return 17
        elif (city1 == 3 and city2 == 4) or (city1 == 4 and city2 == 3):
            return 11
        elif (city1 == 3 and city2 == 5) or (city1 == 5 and city2 == 3):
            return 18
        elif (city1 == 3 and city2 == 6) or (city1 == 6 and city2 == 3):
            return 21
        elif (city1 == 4 and city2 == 5) or (city1 == 5 and city2 == 4):
            return 9
        elif (city1 == 4 and city2 == 6) or (city1 == 6 and city2 == 4):
            return 5
        else:
            return 15

    # complete fitness function
    def fitness(self, dna):
       pass
    """ 
       For each gene in the DNA, there is a 1/mutation_chance chance that it will be 
       switched out with a random character. This ensures diversity in the 
       population, and ensures that is difficult to get stuck in local minima. 
       """

    def mutate(self, dna, random1, random2):
       pass

       """ 
       Slices both dna1 and dna2 into two parts at a random index within their 
       length and merges them. Both keep their initial sublist up to the crossover 
       index, but their ends are swapped. 
       """

    def crossover(self, dna1, dna2, random1, random2):
       pass

# endregion
#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn

                                  #-----0----|---10---|----20---|----30--|---40----|----50---|---60----|
def SearchAlgorithm_Main():       #012345678901234567890123456789013456789012345678901234567890123456789
    # searchAlgo = SearchAlgorithms('.,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,S,#,E,.,#,.',
    #                               [ 2, 15, 2, 100, 60, 35, 30,
    #                                 3, 100, 2, 15, 60, 100, 30,
    #                                 2, 100, 2, 2, 2, 40, 30,
    #                                 2, 2, 100, 100, 3, 15, 30,
    #                                 100,0, 100, 0, 2, 100, 30])

    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                                  [ 0,15, 2,  100, 60, 35, 30, 3
                                          , 100, 2, 15, 60, 100, 30, 2
                                          , 100, 2, 2, 2, 40, 30, 2, 2
                                          , 100, 100, 3, 15, 30, 100, 2
                                          , 100, 0, 2, 100, 30])
    fullPath, path, TotalCost = searchAlgo.AstarManhattanHeuristic()
    print('\n**ASTAR with Manhattan Heuristic ** \nFull Path:' + str(fullPath) + '\nPath is: ' + str(path)
          + '\nTotal Cost: ' + str(TotalCost) + '\n\n')


# endregion

# region KNN_MAIN_FN
'''The dataset classifies tumors into two categories (malignant and benign) (i.e. malignant = 0 and benign = 1)
    contains something like 30 features.
'''


def KNN_Main():
    BC = load_breast_cancer()
    X = []

    for index, row in pd.DataFrame(BC.data, columns=BC.feature_names).iterrows():
        temp = []
        temp.append(row['mean area'])
        temp.append(row['mean compactness'])
        X.append(temp)
    y = pd.Categorical.from_codes(BC.target, BC.target_names)
    y = pd.get_dummies(y, drop_first=True)
    YTemp = []
    for index, row in y.iterrows():
        YTemp.append(row[1])
    y = YTemp;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1024)
    KNN = KNN_Algorithm(7);
    accuracy = KNN.KNN(X_train, X_test, y_train, y_test)
    print("KNN Accuracy: " + str(accuracy))


# endregion

# region Genetic_Algorithm_Main_Fn
def GeneticAlgorithm_Main():
    genetic = GeneticAlgorithm();
    population = genetic.random_population()
    for generation in range(genetic.GENERATIONS):
        # print("Generation %s... Random sample: '%s'" % (generation, population[0]))
        weighted_population = []

        for individual in population:
            fitness_val = genetic.fitness(individual)

            pair = (individual, 1.0 / fitness_val)
            weighted_population.append(pair)
        population = []

        for _ in range(int(genetic.POP_SIZE / 2)):
            ind1 = genetic.weighted_choice(weighted_population)
            ind2 = genetic.weighted_choice(weighted_population)
            ind1, ind2 = genetic.crossover(ind1, ind2, r.random(),r.random())
            population.append(genetic.mutate(ind1,r.random(),r.random()))
            population.append(genetic.mutate(ind2,r.random(),r.random()))

    fittest_string = population[0]
    minimum_fitness = genetic.fitness(population[0])
    for individual in population:
        ind_fitness = genetic.fitness(individual)
    if ind_fitness <= minimum_fitness:
        fittest_string = individual
        minimum_fitness = ind_fitness

    print(fittest_string)
    print(genetic.fitness(fittest_string))


# endregion
######################## MAIN ###########################33
if __name__ == '__main__':
    SearchAlgorithm_Main()
    # KNN_Main()
    # GeneticAlgorithm_Main()
