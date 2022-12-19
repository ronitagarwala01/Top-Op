import numpy as np

"""
List of functions that need to be written (~18 functions):
	-Main driver function (control flow function)

	-Initial population:
		-Member generator

	To be contained within a loop:
		-Evaluation function:
			-Fitness function
			-Dtype packager
		-Selection:
			-Sorting function
			-Elite selector
			-Cull selector
		Pairing:
			-Pairing function
            -Generate pair indicies
		-Crossover:
			-AlternativeRowSwap
			-AlternativeColSwap
			-AltRowAndColSwap
		-Mutation:
			-Mutator function
			-Probability function(?)
"""

"""
Framework:
Generation of new population
    DO NOT MAKE IT MORE COMPLICATED THAN NEEDED

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
    member = np.random.randint(0, 2, size=(nelx, nely))

    return member

def generateInitalPopulation(nelx, nely, numMembers):
    population = []

    for i in range(numMembers):

        population.append(memberGenerator(nelx, nely))

    return population

"""
Wrapped within an iterator:
    Evaluation
        For each member within the population:
            Pass member to fitness function:
                Fitness function (element minimization):
                    Sum the value of each array
                    Return sum
        Pass sum and member to dtype packager:
            Dtype packager:
                dtype = [('member', object), ('sum', int)]
                return np.array(member, sum, dtype=dtype)
"""
def memberAndFitnessPairing(member, fitnessValue):
    memberFitnessTuple = (member, fitnessValue)

    return memberFitnessTuple

def fitnessFunction(member):
    fitnessValue = np.sum(member)

    return fitnessValue

def evaluation(population):
    memberFitnessValuePair = []

    for member in population:
        fitnessValue = fitnessFunction(member)
        memberFitnessValuePair.append(memberAndFitnessPairing(member, fitnessValue))

    return memberFitnessValuePair


"""
    Selection
        Pass member/sum array to sorting function
            Sorting function:
                Sort array of member/sum by sum
                return array
        
        Elite and cull block:
            Pass elite ratio and members to elite selector
                Elite selector:
                    Separate top ratio of memebers from the rest of the population
                    return elite population, common population

            Pass cull ratio and common population to cull function
                Cull function:
                    Given cull ratio, remove bottom % of population
                    return resulting middle population
"""
def calculateNumberToSelect(memberFitnessValuePairs, populationRatio):
    numberToBeSelected = len(memberFitnessValuePairs) * populationRatio
    numberToBeSelected = int(numberToBeSelected)

    if numberToBeSelected % 2 != 0:
        numberToBeSelected += 1

    return numberToBeSelected

def cullSelection(memberFitnessValuePairs, cullRatio=0.4):
    numberToBeCulled = calculateNumberToSelect(memberFitnessValuePairs, cullRatio)

    culledPopulation = memberFitnessValuePairs[:-numberToBeCulled]

    return culledPopulation

def eliteSelection(sortedPairs, eliteRatio=0.01):
    numberOfElite = calculateNumberToSelect(sortedPairs, eliteRatio)
    
    elitePopulation = sortedPairs[:numberOfElite]

    return elitePopulation

def fitnessValueKeyForSort(n):
    return n[1]

def sortMemberFitnessValuePairs(memberFitnessValuePairs):
    sortedScores = sorted(memberFitnessValuePairs, 
                            key=fitnessValueKeyForSort)

    return sortedScores

def selection(sortedPairs):
    # This is included as a catch, so that *if* the population
    # ever does drop this low, the rest of the algorithm still functions
    # Mostly for testing purposes, as the population should never drop this low anyways
    if len(sortedPairs) <= 4:
        return sortedPairs

    culledPopulation = cullSelection(sortedPairs)

    return culledPopulation


"""
    Pairing
        Random pairing of members for crossover
            Pass to pairing function:
                For each member:
                    Pair with a different, randomly picked, member
                    Package both members
                    return list of pairs

        return pairs
"""
def pairIndices(listOfIndices, lengthOfIndexArray):

    if lengthOfIndexArray % 2 != 0:
        randomAdditionalIndex = np.random.randint(0, lengthOfIndexArray)
        listOfIndices = np.append(listOfIndices, randomAdditionalIndex)
        lengthOfIndexArray += 1

    numberOfSubdivisions = lengthOfIndexArray / 2

    pairs = np.split(listOfIndices, numberOfSubdivisions)

    return pairs

def generateRandomIndices(numberOfPopulationMembers):
    indices = np.arange(numberOfPopulationMembers)

    np.random.shuffle(indices)

    return indices

def pairing(culledSortedPopulation):
    numberOfPopulationMembers = len(culledSortedPopulation)
    randomIndices = generateRandomIndices(numberOfPopulationMembers)

    pairedIndices = pairIndices(randomIndices, numberOfPopulationMembers)

    return pairedIndices


"""
    Crossover
        Swapping of genetic information
        3 methods of crossover will be used to start:
            Alternate swapping of rows:
                for each row: 
                    if row index % 2 == 0:
                        swap rows

            Alternate swapping of cols:
                for each row:
                    if col index % 2 == 0:
                        swap rows

            Combination of both:
                for each solution:
                    altRowSwap
                    altColSwap


        Pass to crossover driver:
            List of member pairs

            Call specific crossover method function based on type code

            for pair in list:
                pass to swap function

        return new generation
"""
def alternateRowColSwap(memberPair):
    member1, member2 = alternateRowSwap(memberPair)

    newMemberPair = (member1, member2)

    member1, member2 = alternateColSwap(newMemberPair)

    return member1, member2

def alternateColSwap(memberPair):
    member1 = np.copy(memberPair[0])
    member2 = np.copy(memberPair[1])

    for i in range(member1.shape[1]):
        if i % 2 == 0:
            tempHold = np.copy(member1[:, i])
            member1[:, i] = member2[:, i]
            member2[:, i] = tempHold

    return member1, member2

def alternateRowSwap(memberPair):
    member1 = np.copy(memberPair[0])
    member2 = np.copy(memberPair[1])

    for i in range(member1.shape[0]):
        if i % 2 == 0:
            tempHold = np.copy(member1[i, :])
            member1[i, :] = member2[i, :]
            member2[i, :] = tempHold

    return member1, member2

def crossoverOperationWrapper(solutionPair):
    crossoverSolutions = []

    newSolution1, newSolution2 = alternateRowSwap(solutionPair)
    newSolution3, newSolution4 = alternateColSwap(solutionPair)
    newSolution5, newSolution6 = alternateRowColSwap(solutionPair)

    crossoverSolutions.append(newSolution1)
    crossoverSolutions.append(newSolution2)
    crossoverSolutions.append(newSolution3)
    crossoverSolutions.append(newSolution4)
    crossoverSolutions.append(newSolution5)
    crossoverSolutions.append(newSolution6)

    return crossoverSolutions

def crossover(sortedPopulation, pairIndices):
    # Sorted population
    # For each index pair
    # pairIndices is a list of arrays, each containing two elements
    #   pull members
    #   package (maintain originals)
    #   pass to crossover operations
    #   append new solutions (all 3 possible swaps!) to nextGeneration

    newGeneration = []

    for pair in pairIndices:
        firstIndex = pair[0]
        secondIndex = pair[1]

        solution1 = np.copy(sortedPopulation[firstIndex][0])
        solution2 = np.copy(sortedPopulation[secondIndex][0])
        
        solutionPair = (solution1, solution2)

        newSolutions = crossoverOperationWrapper(solutionPair)
        
        for solution in newSolutions:
            newGeneration.append(solution)

    return newGeneration

"""
    Elite Re-Integration
"""
def extractEliteSolutions(elitePopulation):
    eliteSolutions = []

    for elite in elitePopulation:
        solution = np.copy(elite[0])
        eliteSolutions.append(solution)


    return eliteSolutions

def integrateElite(newPopulation, elitePopulation):
    extractedEliteSolutions = extractEliteSolutions(elitePopulation)

    for solutions in extractedEliteSolutions:
        newPopulation.append(solutions)

    return newPopulation

"""
    Mutation
        Random chance of mutation, given by a probability

        Pass new generation to mutation function:
            For each member:
                Step through each element of solution:
                    For each element:
                        Flip bit based on mutation probability

        return new generation

    ---> Next iteration
"""

def shouldMutate():
    probability = [0.6, 0.4]
    boolInt = [0, 1]
    return np.random.choice(boolInt, p=probability)


def mutateMember(solution):
    unraveledSolution = np.ravel(solution)
    mutatedSolution = []

    for element in unraveledSolution:
        if shouldMutate():
            element = int(np.absolute(element - 1))
            mutatedSolution.append(element)
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


"""
    Population Control
        Pass crossed members to population control function
        if exceeds population limit (1000 for now):
            evaluate
            sort
            take first 1000 members from crossed and sorted list
        
        return population for mutation
"""
def populationControl(crossedMembers, maxPopLimit):
    if len(crossedMembers) <= maxPopLimit:
        return crossedMembers

    evaluatedCrossedMembers = evaluation(crossedMembers)
    sortedEvalCrossedMembers = sortMemberFitnessValuePairs(evaluatedCrossedMembers)
    print("Top fitness value: ", sortedEvalCrossedMembers[0][1])

    culledPopulation = sortedEvalCrossedMembers[:maxPopLimit]

    newGeneration = [solution for solution, fitness in culledPopulation]

    return newGeneration

"""
    Main Wrapper Function
"""

def convergenceTest(population, goal):
    firstMember = population[0]

    if np.sum(firstMember) == 0:
        return True
    return False

def mainWrapper(nelx, nely, numPop, numIterations):
    newPopulation = generateInitalPopulation(nelx, nely, numPop)

    for x in range(numIterations):
        print("Iteration:", x)

        memberFitnessValuePairs = evaluation(newPopulation)

        sortedPopulation = sortMemberFitnessValuePairs(memberFitnessValuePairs)

        culledPopulation = selection(sortedPopulation)
        elitePopulation = eliteSelection(sortedPopulation)

        pairIndices = pairing(culledPopulation)

        crossedPopulation = crossover(culledPopulation, pairIndices)

        integratedPopulation = integrateElite(crossedPopulation, elitePopulation)

        controlledPopulation = populationControl(integratedPopulation, 700)

        if convergenceTest(controlledPopulation, 0):
            print('"Converged"')
            print(controlledPopulation[0])
            return controlledPopulation


        mutatedPopulation = mutation(controlledPopulation)
        
        newPopulation = mutatedPopulation

        print(len(newPopulation))

    return newPopulation





"""
What follows is Code for testing, this will be moved eventually, or removed
"""
# Main Wrapper Testing
singleIteration = mainWrapper(6, 6, 10, 100)
# print(singleIteration)



# Population Generation Testing
# testMember = memberGenerator(2, 2)
# print("Test Member: ", testMember)

# testPopulation = generateInitalPopulation(2, 2, 2)
# print("Test Population: ", testPopulation)




# Evaluation Testing
# testPopulation = generateInitalPopulation(2, 2, 2)

# print(testPopulation)

# testFitnessValue = fitnessFunction(testPopulation[0])
# print("\nfitnessValue test: ", testFitnessValue)

# testFitnessValueMemberPair = memberAndFitnessPairing(testPopulation[0], testFitnessValue)
# print("\nMember and fitnessValue pair test: ", testFitnessValueMemberPair)

# fullEvaluationTestReturn = evaluation(testPopulation)
# print("\nFrom evaluation function: ", fullEvaluationTestReturn)



# Selection Testing
def generateToEvaluation(nelx, nely, numPop):
    testPopulation = generateInitalPopulation(nelx, nely, numPop)

    testEvaluated = evaluation(testPopulation)

    return testEvaluated

# unsortedTestPairs = generateToEvaluation(2, 2, 8)
# print("Unsorted pairs: ")
# print(unsortedTestPairs)

# sortedTestPairs = sortMemberFitnessValuePairs(unsortedTestPairs)

# print("\nSorted pairs: ")
# print(sortedTestPairs)

# testElitePopulation = eliteSelection(sortedTestPairs, 0.2)
# print(testElitePopulation)

# testExtractedElite = extractEliteSolutions(testElitePopulation)
# print(testExtractedElite)


# print("Length of pairs: ", len(sortedTestPairs))

# xs = [1, 2, 3, 4, 5, 6]
# ys = xs[:-2]

# print(ys)

# testNumberToCull = calculateNumberToCull(sortedTestPairs, 0.2)
# print("Expected: 2; Actual: ", testNumberToCull)

# print("Before Cull: ", sortedTestPairs)

# culledPop = cullSelection(sortedTestPairs)

# print("After Cull: ", culledPop)




# Pairing Testing
def generateToSelection(nelx, nely, numPop):
    testPopulation = generateToEvaluation(nelx, nely, numPop)

    selectedPopulation = selection(testPopulation)

    return selectedPopulation


# Testing numpy's shuffle. This will be used to determine the pairs
# xs = np.arange(6)
# np.random.shuffle(xs)

# ys = np.split(xs, 3)
# print(ys)
# print(ys[0])
# print(ys[0][0])


# testShuffledIndices = generateRandomIndices(9)
# print(testShuffledIndices)

# testPairs = pairIndices(testShuffledIndices, 9)
# print(testPairs)

# testPopulation = generateToSelection(2, 2, 8)
# print(testPopulation)
# print(len(testPopulation))

# testPairingPipeline = pairing(testPopulation)
# print(testPairingPipeline[0])

# print(type(testPairingPipeline[0]))





# Crossover testing
def generateToPairing(nelx, nely, numPop):
    culledPopulation = generateToSelection(nelx, nely, numPop)

    pairedPopulation = pairing(culledPopulation)

    return culledPopulation, pairedPopulation

# ones = np.random.randint(0, 2, (3, 6))

# print(ones)
# print(ones.shape)
# print(ones.shape[1])

# print(ones[:, 0])

def rowSwapTest():    
    array1 = np.arange(1, 10)
    array1 = array1.reshape(3, 3)

    array2 = np.arange(10, 19)
    array2 = array2.reshape(3, 3)

    print(array1)
    print(array2)
    print("")

    testPair = (array1, array2)

    testArray1, testArray2 = alternateRowSwap(testPair)

    print(testArray1)
    print(testArray2)

def colSwapTest():    
    array1 = np.arange(1, 10)
    array1 = array1.reshape(3, 3)

    array2 = np.arange(10, 19)
    array2 = array2.reshape(3, 3)

    print(array1)
    print(array2)
    print("")

    testPair = (array1, array2)

    testArray1, testArray2 = alternateColSwap(testPair)

    print(testArray1)
    print(testArray2)

def rowColSwapTest():    
    array1 = np.arange(1, 10)
    array1 = array1.reshape(3, 3)

    array2 = np.arange(10, 19)
    array2 = array2.reshape(3, 3)

    print(array1)
    print(array2)
    print("")

    testPair = (array1, array2)

    testArray1, testArray2 = alternateRowColSwap(testPair)

    print(testArray1)
    print(testArray2)

# testSelected, testPairs = generateToPairing(3, 3, 2)

# print(testSelected)
# print(testPairs)

# testNextGeneration = crossover(testSelected, testPairs)

# print(testNextGeneration)



def generateToCrossover(nelx, nely, numPop):
    testSelect, testPairs = generateToPairing(nelx, nely, numPop)

    newGeneration = crossover(testSelect, testPairs)

    return newGeneration



# print(shouldMutate())

# totals = [0, 0]

# for i in range(100):
#     totals[shouldMutate()] += 1

# print(totals)

# testArray = np.array([0, 0])
# print(testArray)

# testArray[0] = 1
# print(testArray)

# testArray = np.ones((3, 3), dtype=np.int32)
# print(testArray)

# testArray = mutateMember(testArray)
# print(testArray)

# testNewGen = generateToCrossover(3, 3, 2)
# print(testNewGen)

# testMember = generateInitalPopulation(4, 4, 1)
# print(testMember)

# testMutated = mutation(testMember)
# print(testMutated)


