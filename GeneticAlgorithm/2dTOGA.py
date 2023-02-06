from time import perf_counter
import numpy as np
import pandas as pd
from fenicsGA2D import *
import math
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.ndimage import label

"""
Framework:
Generation of new population
    To wrapper function:
        Given dimensions and number of members to generate
        Call member generator:
            Member generator:
                Do:
                    Create array of given dimensions
                    Randomly populate each index of array with either 0 or 1
                Return array
"""
def memberGenerator(nelx, nely):
    member = np.random.choice([1e-5, 1], p = [0.5, 0.5], size=(nelx, nely))
    # member = np.ones((nelx, nely))

    return member


def generateInitalPopulation(nelx, nely, numMembers):
    population = []

    for i in range(numMembers):

        population.append(memberGenerator(nelx, nely))

    return population

global penaltyExponent
penaltyExponent = -5

"""
    Evaluation
"""
def memberAndFitnessPairing(member, fitnessValue):
    memberFitnessTuple = (member, fitnessValue)

    return memberFitnessTuple


def fitnessFunction(member, nelx, nely):
    fitnessValue = np.sum(member)

    roundedMember = np.round(member)
    labeled_array, numFeatures = label(roundedMember)

    fitnessValue *= numFeatures

    C_max = 1000.0
    # S_max = 100000000.0

    # if (numFeatures >= 0) and (numFeatures <= 3):
    if numFeatures >= 1 and numFeatures < 5:
        stress, compliance = solveConstraints(member)

        if math.isnan(compliance):
            return np.inf

        if compliance >= C_max:
            fitnessValue += (math.exp(penaltyExponent) * (compliance - C_max))

        print("Compliance: ", compliance)

    return fitnessValue


def evaluation(population, nelx, nely):
    memberFitnessValuePair = []

    memberNum = 1

    for member in population:
        # print(memberNum); memberNum += 1

        fitnessValue = fitnessFunction(member, nelx, nely)
        memberFitnessValuePair.append(memberAndFitnessPairing(member, fitnessValue))

    return memberFitnessValuePair





"""
    Selection
"""
def fitnessValueKeyForSort(n):
    return n[1]

def sortMemberFitnessValuePairs(memberFitnessValuePairs):
    sortedScores = sorted(memberFitnessValuePairs, 
                            key=fitnessValueKeyForSort, reverse=False)

    return sortedScores

def probablilityToSelect(member):
    memberSolution = member[0]

    memberDimensions = memberSolution.shape

    obj = np.sum(memberSolution)
    objMax = memberDimensions[0] * memberDimensions[1]

    p = 1 - (obj / objMax)

    return p

def shouldSelect(member):
    p = probablilityToSelect(member)

    probabilities = [1 - p, p]
    bools = [0, 1]

    return np.random.choice(bools, p = probabilities)


def selection(sortedPairs, numToSelect, numElite):

    if len(sortedPairs) <= 4:
        return sortedPairs

    if numElite >= len(sortedPairs):
        return sortedPairs

    numSelected = numElite
    numToSelect -= numElite

    selectedPopulation = sortedPairs[:numElite]

    toSelect = sortedPairs[numElite:] 

    np.random.shuffle(toSelect)

    while(numToSelect > 0):
        if len(toSelect) < numToSelect:
            break

        np.random.shuffle(toSelect)

        for i in range(0, numToSelect):
            if shouldSelect(toSelect[i]):
                selected = toSelect.pop(i)
                selectedPopulation.append(selected)
                numToSelect -= 1

    return selectedPopulation
    





"""
    Crossover
"""
def swapRandomRowBlocks(Individuals):

    Individual1, Individual2 = Individuals
    Child1 = np.empty_like(Individual1)

    Row = np.random.randint(0,Individual1.shape[0])

    Child1[0:Row, :] = Individual1[0:Row, :]
    Child1[Row:, :] = Individual2[Row:, :]

    Child2 = np.empty_like(Individual1)

    Child2[0:Row, :] = Individual2[0:Row, :]
    Child2[Row:, :] = Individual1[Row:, :]

    return Child1, Child2

def swapRandomColBlocks(Individuals):

    Individual1, Individual2 = Individuals
    Child1 = np.empty_like(Individual1)

    Col = np.random.randint(0,Individual1.shape[1])

    Child1[:, 0:Col] = Individual1[:, 0:Col]
    Child1[:, Col:] = Individual2[:, Col:]

    Child2 = np.empty_like(Individual1)

    Child2[:, 0:Col] = Individual2[:, 0:Col]
    Child2[:, Col:] = Individual1[:, Col:]

    return Child1, Child2

def swapRandomRowColBlocks(Individuals):

    rowCross = swapRandomRowBlocks(Individuals)
    rowColCross = swapRandomColBlocks(rowCross)

    return rowColCross

def crossoverBooleanNot(solutionPair):
    Individual1, Individual2 = solutionPair

    child1 = np.logical_not(Individual1) * 1
    child2 = np.logical_not(Individual2) * 1

    return child1, child2

def crossoverBooleanAnd(solutionPair):
    Individual1, Individual2 = solutionPair

    child = np.logical_and(Individual1, Individual2) * 1
    return child

def crossoverBooleanOr(solutionPair):
    Individual1, Individual2 = solutionPair

    child = np.logical_or(Individual1, Individual2) * 1
    return child

def crossoverBooleanXor(solutionPair):
    Individual1, Individual2 = solutionPair

    child = np.logical_xor(Individual1, Individual2) * 1
    return child



def crossoverOperationWrapper(solutionPair, alternate=1):
    crossoverSolutions = []

    newSolution1, newSolution2 = swapRandomRowBlocks(solutionPair)
    newSolution3, newSolution4 = swapRandomColBlocks(solutionPair)
    newSolution5, newSolution6 = swapRandomRowColBlocks(solutionPair)
    # newSolution7, newSolution8 = crossoverBooleanNot(solutionPair)
    newSolution9               = crossoverBooleanAnd(solutionPair)
    # newSolution10              = crossoverBooleanOr(solutionPair)
    # newSolution11              = crossoverBooleanOr(solutionPair)

    crossoverSolutions.append(newSolution1)
    crossoverSolutions.append(newSolution2)
    crossoverSolutions.append(newSolution3)
    crossoverSolutions.append(newSolution4)
    crossoverSolutions.append(newSolution5)
    crossoverSolutions.append(newSolution6)
    # crossoverSolutions.append(newSolution7)
    # crossoverSolutions.append(newSolution8)
    crossoverSolutions.append(newSolution9)
    # crossoverSolutions.append(newSolution10)
    # crossoverSolutions.append(newSolution11)

    return crossoverSolutions

def crossover(shuffledPopulation, alternate=0):
    # Takes a shuffled population, we are just crossing each member along the array

    newMembers = []
    numMembers = len(shuffledPopulation)

    for i in range(numMembers):
        if i + 1 >= numMembers:
            memberPair = (shuffledPopulation[i], shuffledPopulation[0])
        else:
            memberPair = (shuffledPopulation[i], shuffledPopulation[i + 1])

        newSolutions = crossoverOperationWrapper(memberPair, alternate)
        
        for solution in newSolutions:
            newMembers.append(solution)

    return newMembers





"""
    Mutation
"""
def shouldMutate():
    probability = mutationProbability
    boolInt = [0, 1]
    return np.random.choice(boolInt, p=probability)

def shouldMutateVar(probabilities):
    boolInt = [0, 1]
    return np.random.choice(boolInt, p=probabilities)


def mutateMember(solution):
    unraveledSolution = np.ravel(solution)
    mutatedSolution = []

    for element in unraveledSolution:
        if shouldMutate():
            if element == 1:
                mutatedSolution.append(1e-5)
            else:
                mutatedSolution.append(1)
        else:
            mutatedSolution.append(element)

    mutatedSolution = np.reshape(mutatedSolution, newshape=solution.shape)

    return mutatedSolution

def mutation(newGeneration):
    mutatedPopulation = []

    for member in newGeneration:
        mutatedMember = mutateMember(member)
        mutatedPopulation.append(mutatedMember)

    return  mutatedPopulation

def materialAroundElement(member, x, y):
    xMax = member.shape[1]
    yMax = member.shape[0]

    xi, xj, yi, yj = x-1, x+1, y-1, y+1

    if x == 0: 
        xi = x
    elif x == xMax:
        xj = x
    
    if y == 0:
        yi = y
    elif y == yMax:
        yj = y

    surroundingMaterial = np.sum(member[yi:yj+1, xi:xj+1])

    return surroundingMaterial


# def adaptiveMutation(member):
#     indices, surroundingMass = [], []

#     for i in range(member.shape[0]):
#         for j in range(member.shape[1]):
#             index = (i, j)
#             indices.append(index)

#     for x, y in indices:
#         mass = materialAroundElement(member, x, y)
#         surroundingMass.append(mass)

#     surroundingMass = np.array(surroundingMass)
#     flatMember = np.ravel(member)

#     mutatedSolution = []

#     for x in range(len(flatMember)):
#         p = surroundingMass[x] / 10
#         probabilities = [p, 1 - p]

#         if surroundingMass[x] == 1 and flatMember[x] == 1:
#             mutatedSolution.append(1e-5)
#         elif shouldMutateVar(probabilities):
#             if flatMember[x] == 1:
#                 mutatedSolution.append(1e-5)
#             else:
#                 mutatedSolution.append(1)
#         else:
#             mutatedSolution.append(flatMember[x])

#     mutatedSolution = np.reshape(mutatedSolution, newshape=member.shape)

#     return mutatedSolution


def adaptiveMutation(member):
    surroundingMass = 0


    for y in range(member.shape[0]):
        for x in range(member.shape[1]):
            surroundingMass = materialAroundElement(member, x, y)

            if member[y, x] == 1.e-05 and surroundingMass >= 5:
                member[y, x] = 1.e+00
                continue

            p = surroundingMass / 10

            if shouldMutateVar([p, 1-p]):
                if member[y, x] == 1.e+00:
                    member[y, x] = 1.e-05
                else:
                    member[y, x] = 1.e+00

    return member


def adaptiveMutationPipeline(generation):
    newGeneration = []

    for member in generation:
        mutatedMember = adaptiveMutation(member)
        newGeneration.append(mutatedMember)

    return newGeneration


"""
    Side Functions
"""
def fitnessAverage(evaluatedMembers, listOfAverages = []):
    fitnessValues = []
    for solution, fitness in evaluatedMembers:
        fitnessValues.append(fitness)

    return np.average(fitnessValues)


def convergenceTest(population, goal):
    firstMember = population[0]

    print(("Top Performing Fitness Value: ", firstMember[1]))

    if np.sum(firstMember[1]) == goal:
        return True
    return False

def extractSolutions(solutionPairs):
    solutions = []

    for solution, fitness in solutionPairs:
        solutions.append(solution)

    return solutions


def prune(solutionPairs):
    prunedMembers = []

    for solution, fitness in solutionPairs:
        if fitness != np.inf:
            pair = (solution, fitness)
            prunedMembers.append(pair)

    return prunedMembers



def isConnected(member, x, y):
    xMax = member.shape[1] - 1
    yMax = member.shape[0] - 1

    #   1 
    # 0   3   
    #   2   
    adjX = [x-1, x, x, x+1]
    adjY = [y, y-1, y+1, y]

    # No, this doesn't make sense to me,
    # but, it is what sets diag to be intuitive
    # for this format:
        # 0   1
        # 2   3
    diagX = [x-1, x+1, x-1, x+1]
    diagY = [y-1, y-1, y+1, y+1]

    canCheckAdj = [1, 1, 1, 1]
    canCheckDiag = [1, 1, 1, 1]

    connected = [0, 0, 0, 0]
    diag = [0, 0, 0, 0]

    if x == 0:
        canCheckAdj[0] = 0
        canCheckDiag[0], canCheckDiag[2] = 0, 0
    elif x == xMax:
        canCheckAdj[3] = 0
        canCheckDiag[1], canCheckDiag[3] = 0, 0

    if y == 0:
        canCheckAdj[1] = 0
        canCheckDiag[0], canCheckDiag[1] = 0, 0
    elif y == yMax:
        canCheckAdj[2] = 0
        canCheckDiag[2], canCheckDiag[3] = 0, 0


    for i in range(len(canCheckAdj)):
        if canCheckAdj[i]:
            # print('adj: ', i)
            # print(adjX[i], adjY[i])
            if member[adjY[i], adjX[i]] == 1.e+00:
                connected[i] = 1

        if canCheckDiag[i]:
            # print('diag:', i)
            if member[diagY[i], diagX[i]] == 1.e+00:
                diag[i] = 1

    numConnectedAdj = np.sum(connected)
    numDiag = np.sum(diag)
    numSurroundingElements = numConnectedAdj + numDiag
    
    # if (member[y, x] == 1.e-05) and (numSurroundingElements >= 7):
    #     member[y, x] = 1.e+00
    #     return member

    if numSurroundingElements == 0:
        member[y, x] = 1.e-05
        return member

    # I am not entirely sure which of this conditionals will provide the best result.
    # It seems as though it is something that matters and doesn't matter at the same time.

    # if (numConnectedAdj >= 0) and (numConnectedAdj <= 2):
    if numConnectedAdj >= 0 and numConnectedAdj < 2:
        for i in range(len(diag)):
            if diag[i]:
                adj1 = member[diagY[i], x]

                if adj1 == 1.e+00:
                    member[y, diagX[i]] = 1.e+00
                else:
                    member[diagY[i], x] = 1.e+00

    return member


def connectivityConstraint(population):
    patchedPop = []

    for i in range(int(len(population) / 3)):
        member = population[i]
        for y in range(member.shape[0]):
            for x in range(member.shape[1]):
                # if member[y, x] == 1.e+00:
                member = isConnected(member, x, y)
        patchedPop.append(member)

    patchedPop = np.array(patchedPop)

    return patchedPop

"""
    Main!
"""
def mainWrapper(nelx, nely, numPop, numIterations):
    nIterCurrent, nFailures = 0, 0
    newPopulation = generateInitalPopulation(nelx, nely, numPop)

    for x in range(numIterations):

        if x % 5 == 0:
            global penaltyExponent
            penaltyExponent += 1

        if x % 10 == 0:
            for i in range(10):
                newPopulation.append(np.ones(shape=(nelx, nely)))

        nIterCurrent += 1
        print("Iteration:", x)

        # Duplication
        toCross = np.copy(newPopulation)
        toMutate = np.copy(newPopulation)
        toMutateAdaptive = np.copy(newPopulation)
        
        # This will be passed raw to the crossover algorithm
        np.random.shuffle(toCross)

        # Crossover & Mutation
        newCrossedMembers = crossover(toCross)
        newMutatedMembers = mutation(toMutate)
        newAdaptiveMutated = adaptiveMutationPipeline(toMutateAdaptive)
        
        for crossed in newCrossedMembers:
            newPopulation.append(crossed)

        for mutated in newMutatedMembers:
            newPopulation.append(mutated)

        for mutated in newAdaptiveMutated:
            newPopulation.append(mutated)

        if x % 4 == 0 and x >= 2:
            newPopulation = connectivityConstraint(newPopulation)
        
        # newPopulation = connectivityConstraint(newPopulation)

        # Evaluation
        memberFitnessValuePairs = evaluation(newPopulation, nelx, nely)
        print("Avg fitness: ", fitnessAverage(memberFitnessValuePairs, []))

        # Pruning
        # This removes all solutions from the infeasible region
        memberFitnessValuePairs = prune(memberFitnessValuePairs)

        # print("Number of solutions: ", len(memberFitnessValuePairs))

        if len(memberFitnessValuePairs) == 0:
            print("Failed")
            nFailures += 1
            newPopulation = generateInitalPopulation(nelx, nely, numPop )
            continue

        # Selection
        sortedPopulation = sortMemberFitnessValuePairs(memberFitnessValuePairs)

        # Selection takes pop, numToSelect, and numElite
        # numToSelect is basically the population cap
        # pop, numToKeep, numElite
        selectedPopulation = selection(sortedPopulation, 400, 100)

        print(selectedPopulation[0])

        if convergenceTest(selectedPopulation, 0):
            print('"Converged"')
            print(selectedPopulation[0])
            return selectedPopulation[0], nIterCurrent

        newPopulation = extractSolutions(selectedPopulation)

        plt.style.use('_mpl-gallery-nogrid')

        fig, ax = plt.subplots()
        ax.imshow(newPopulation[0])
        plt.show(block=True)

        fileName = str(nelx) + "x" + str(nely) + "i_" + str(x)

        plt.savefig(fileName)

        plt.close()

    return newPopulation, nIterCurrent







"""
What follows is Code for testing, this will be moved eventually or removed
"""
# Main Wrapper Testing
mutationProbability = [0.95, 0.05]

def benchmarking(start, end, runsPerDim):
    testIterations, testAverageIterations = [], []
    testTime, testAverageTime = [], []
    testTimeForEachIteration = []

    for x in range(start, end):
        testIterations.append([])
        testTime.append([])

        for i in range(runsPerDim):
            startTime = perf_counter()
            _, numIterationsTaken = mainWrapper(x, x, 200, 10000)
            endTime = perf_counter()

            time = (endTime - startTime)
            testTime[x - start].append(time)

            testIterations[x - start].append(numIterationsTaken)
        
        testAverageIterations.append(np.average(testIterations[x - start]))
        testAverageTime.append(np.average(testTime[x - start]))

    for x in range(len(testAverageTime)):
        testTimeForEachIteration.append(testAverageTime[x] / testAverageIterations[x])

    print("\nFull array of iterations: ", testIterations)
    print("\nFull array of times: ", testTime)

    print("\nAverage numbers of iterations: ", testAverageIterations)
    print("Average numbers of times: ", testAverageTime)

    print("Estimate time between for each iteration: ", testTimeForEachIteration)

    return

# benchmarking(3, 9, 10)

nelx, nely, numPop, numIter = 30, 15, 400, 40

startTime = perf_counter()
finalSolutions, _ = mainWrapper(nelx, nely, numPop, numIter)

endTime = perf_counter()

time = endTime - startTime


s = time % 60
m = time / 60
h = m / 60
m = m % 60

print("Hrs: ", h, "Min: ", m, "Sec: ", s)



# plt.style.use('_mpl-gallery-nogrid')

# fig, ax = plt.subplots()
# ax.imshow(finalSolutions)
# plt.show(block=True)

# fileName = str(nelx) + "x" + str(nely) + "i_" + "final"

# plt.savefig(fileName)