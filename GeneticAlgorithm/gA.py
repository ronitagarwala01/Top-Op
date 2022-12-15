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
def calculateNumberToCull(memberFitnessValuePairs, cullRatio):
    numberToBeCulled = len(memberFitnessValuePairs) * cullRatio
    numberToBeCulled = int(numberToBeCulled)

    if numberToBeCulled % 2 != 0:
        numberToBeCulled += 1

    return numberToBeCulled

def cullSelection(memberFitnessValuePairs, cullRatio=0.2):
    numberToBeCulled = calculateNumberToCull(memberFitnessValuePairs, cullRatio)

    culledPopulation = memberFitnessValuePairs[:-numberToBeCulled]

    return culledPopulation


def fitnessValueKeyForSort(n):
    return n[1]

def sortMemberFitnessValuePairs(memberFitnessValuePairs):
    sortedScores = sorted(memberFitnessValuePairs, 
                            key=fitnessValueKeyForSort,
                            reverse=True)

    return sortedScores

def selection(memberFitnessValuePairs):
    sortedPairs = sortMemberFitnessValuePairs(memberFitnessValuePairs)

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
            Type code to indicate which crossover method to use

            Call specific crossover method function based on type code

        return new generation
"""

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


"""
What follows is Code for testing, this will be moved eventually, or removed
"""
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

# print("Length of pairs: ", len(sortedTestPairs))

# xs = [1, 2, 3, 4, 5, 6]
# ys = xs[:-2]

# print(ys)

# testNumberToCull = calculateNumberToCull(sortedTestPairs, 0.2)
# print("Expected: 2; Actual: ", testNumberToCull)

# print("Before Cull: ", sortedTestPairs)

# culledPop = cullSelection(sortedTestPairs)

# print("After Cull: ", culledPop)



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



testPopulation = generateToSelection(2, 2, 8)
print(testPopulation)
print(len(testPopulation))

testPairingPipeline = pairing(testPopulation)
print(testPairingPipeline)

