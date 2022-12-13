import numpy as np

# Very first chunk of code
# If you want to know what the "grand idea" is
# This scratches the surface
def illustrateConcept():
    # X = np.zeros((5,5), dtype= np.int8)
    # print(X)

    dSpace = np.zeros((5,5), dtype= np.int8)
    bounding = np.zeros((5,5), dtype= np.int8)

    bounding[0][0] = 1
    bounding[4][0] = 1
    bounding[2][4] = 2

    print()
    print("Our design space:")
    print(dSpace)
    print()
    print("Our bounding conditions: ")
    print(bounding)

    print()
    print('An "ideal" solution:')
    dSpace[0][0:3] = 1
    dSpace[1][1:4] = 1
    dSpace[2][3:5] = 1
    dSpace[3][1:4] = 1
    dSpace[4][0:3] = 1
    print(dSpace)

    print() 
    chromosome = np.ravel(dSpace)
    print("Chromosome of this solution: ", chromosome)
    print()

def generateSolutionsTest():
    randomSolution1 = np.random.randint(0, 2, size=(3,3))
    randomSolution2 = np.random.randint(0, 2, size=(3,3))

    print(randomSolution1)
    print()
    print(randomSolution2)
    print()

    c1 = np.ravel(randomSolution1)
    c2 = np.ravel(randomSolution2)

    print(c1)
    print(c2)

    cross = np.random.randint(0, 2, size=(1, 3))
    print("A given random crossover: ", cross)


    crossOver(cross, c1, c2)

    n1, n2 = crossOver(cross, c1, c2)

    print()
    print("Performing crossover: ")
    print(cross)
    print(c1)
    print(c2)
    print()
    print(n1)
    print(n2)

# The function below does *not* work with the 'crossOver' function
# This is because np.random.randint() generates a numpy array with a different
# shape than one self-defined.... I think....
# Dunno. The fix is easy if this code is needed again, but why would it be?
def definedCrossOverExample():
    c1 = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0])
    c2 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1])
    cross = np.array([1,0,0])

    n1, n2 = crossOver(cross, c1, c2)

    print()
    print("Performing crossover: ")
    print(cross)
    print(c1)
    print(c2)
    print()
    print(n1)
    print(n2)

def testingCrossoverWithANDSelection(topPerformingSolutions):
    top1 = topPerformingSolutions[0]
    top2 = topPerformingSolutions[1]

    new1, new2 = crossOver(top1, top2)

    print("Top1: ", top1, "\nTop2: ", top2, "\nNew1: ", new1, "\nNew2: ", new2)

# Core-Functionality
# Depreciated and un-used in iterations
# For conceptual purposes and testing
def crossOver(crossGene, chromo1, chromo2):
    new1 = np.copy(chromo1)
    new2 = np.copy(chromo2)

    j = 0
    for i in range(len(chromo1)):
        if j == len(crossGene[0]):
            j = 0

        if crossGene[0][j] == 1:                    # to use literally everything else
        # if crossGene[j]:                           # to use definedCrossOverExample()
            new1[i], new2[i] = new2[i], new1[i]
        
        j += 1

    return new1, new2


# 
#   Beginning of actual functionality
# 

# Core-Functionality
def generateSolution(nelx, nely, randomCross=1):
    newSolution = np.random.randint(0, 2, size=(nelx, nely))
    newChromosome = np.ravel(newSolution)


    if randomCross:
        newCross = np.random.randint(0, 2, size=(1, nely))
    else:
        newCross = []
        for i in range(nely):
            if i % 2 == 0:
                newCross.append(0)
            else:
                newCross.append(1)

        newCross = np.array(newCross)

    return newSolution, newChromosome, newCross


# Part of the fitness function
def sumMass(chromosome):
    return np.sum(chromosome)


def generatePopulation(nelx, nely, numPop):
    populationMembers = []

    for i in range(numPop):
        # solution, gene/chromosome, crossgene
        xs, xg, xc = generateSolution(nelx, nely)

        populationMember = (xs, xg, xc)
        populationMembers.append(populationMember)


    dtype = [('solution', object), ('chromosome', object), ('crossGene', object)]
    populationMembers = np.array(populationMembers, dtype=dtype)


    # I also advise printing populationMembers out
    # into the console, just to see how the list actually
    # is represented in code.
    # Functionally, it does as is intended. So who cares?
    return populationMembers


def fitnessEvaluation(population, fitnessFunction, nTop):
    popScores = []
    dtype = [('populationMember', object), ('fitnessScore', int)]
    

    for member in population:
        fitnessScore = fitnessFunction(member[1])
        memberAndScore = (member, fitnessScore)
        popScores.append(memberAndScore)

    popScores = sorted(popScores, key=lambda t: t[1])

    popScores = np.array(popScores, dtype=dtype)


    topMembers = popScores[:nTop]
    bottomMembers = popScores[nTop:]

    return topMembers, bottomMembers


def extractSolutions(fitnessScores):
    topSolutions = []

    for solution in fitnessScores:
        topSolutions.append(solution[0])

    return topSolutions


# Crossover gene problems:
#   We have a cGene for everysingle solution
#   Not all of them are the same
#   Since we are using random cGene's we want to preserve some semblance of genes
#   We also want to explore the cGene space to see if there are any cGene's that are high performers...
# 
#   Possible solutions:
#       Using one parent's cGene at random
#       Bit masking to create a unique cGene for that pairing (AND [more than likely])
#       Defaulting to the first parent's cGene regardless(?...)
#       Defaulting to the parent's cGene with the best fitness value (fuck this one, stretch goal)

# Using one parent's cGene at random
# Dumb, but it works, and makes the code neater
def crossGeneSelectionRandom(solution1, solution2):
    parent1 = np.random.randint(0, 2)

    gene1 = solution1[2]
    gene2 = solution2[2]

    if parent1:
        return gene1
    else:
        return gene2

# This simply performs an AND operation on the genes (bitmasking, essentially)
def crossGeneMaskAND(solution1, solution2):
    newGene = []
    gene1 = solution1[2][0]
    gene2 = solution2[2][0]

    print(gene1)
    print(gene1.shape)


    for i in range(len(gene1)):
        if gene1[i] == gene2[i]:
            newGene.append(gene1[i])
        else:
            newGene.append(0)

    newGene = np.array(newGene)

    return newGene

# For the sake of having a function to call
# (it will help the overall implementation, trust me)
# and yes, it does take two parameters, even though it doesn't even use the second
# and no, I don't care right now.           # I am a menace that cannot be stopped
def extractFirstCrossGene(soltuion1, solution2):
    return soltuion1[2]


def repackageSolution(newChromo, crossGene, shape):
    newSolution = np.reshape(newChromo, newshape=shape)

    return (newSolution, newChromo, crossGene)


def numElite(eliteRatio, numPop):
    numE = numPop * eliteRatio

    numE = int(numE)
    
    if numE % 2 != 0:
        numE += 1

    return numE


def calculateCullIndex(lowerPop, cullRatio):
    lowerLen = len(lowerPop)

    cullIndex = int(lowerLen * cullRatio)

    return cullIndex


def selection(elitePop, bottomPop, cullIndex):
    selectionPairs = []

    # This should generate n = (len(elitePop) - 1) pairings, which should result in 2n new solutions
    for i in range(len(elitePop) - 1):
        selectionPairs.append((elitePop[i], elitePop[i + 1]))
    
    newBottom = bottomPop[:cullIndex]

    for i in range(len(newBottom) - 1):
        selectionPairs.append((newBottom[i], newBottom[i + 1]))

    
    return selectionPairs


# Core-Functionality
# New version, it allows for different cGene selection functions
# Expanded to:
#   Take full solutions instead of just the chromosomes
#   Return 2 new solutions
def crossOver(solution1, solution2, crossSelection=crossGeneMaskAND):
    crossGene = crossSelection(solution1, solution2)
    shape = solution1[0].shape

    newChromo1 = np.copy(solution1[1])
    newChromo2 = np.copy(solution2[1])

    j = 0
    for i in range(len(solution1[1])):
        if j == len(crossGene):
            j = 0

        if crossGene[j] == 1:          
            newChromo1[i], newChromo2[i] = newChromo2[i], newChromo1[i]
        
        j += 1

    newSolution1 = repackageSolution(newChromo1, crossGene, shape)
    newSolution2 = repackageSolution(newChromo2, crossGene, shape)

    return newSolution1, newSolution2


def crossOverWrapper(selectedPop):
    newGeneration = []

    for pair in selectedPop:
        nS1, nS2 = crossOver(pair[0], pair[1], crossSelection=extractFirstCrossGene)

        newGeneration.append(nS1)

        newGeneration.append(nS2)

    dtype = [('solution', object), ('chromosome', object), ('crossGene', object)]
    newGeneration = np.array(newGeneration, dtype=dtype)


    return newGeneration


def mainWrapper(nelx, nely, numIntial, eliteRatio=0.4, cullRatio=0.5, iterations=10):
    # Initial population
    newGeneration = generatePopulation(nelx, nely, numIntial)

    while(iterations > 0):

        numPop = len(newGeneration)
        numOfElite = numElite(eliteRatio, numPop)

        # Core of the genetic algorithm
        # To be contained within a iterator/wrapper
        topFitnessScores, bottomFitnessScores = fitnessEvaluation(newGeneration, sumMass, numOfElite)
        
        eliteSolutions = extractSolutions(topFitnessScores)
        lowerSolutions = extractSolutions(bottomFitnessScores)

        cullIndex = calculateCullIndex(lowerSolutions, cullRatio)

        selectedPop = selection(eliteSolutions, lowerSolutions, cullIndex)

        newGeneration = crossOverWrapper(selectedPop)

        print(newGeneration)

        print("Next generation!")

        iterations -= 1


mainWrapper(3, 3, 10, iterations=2)


# To-Do:  
#   Selection -> Crossover -> Return new population
#   Retaining some members of the original population
#       Retaining Elite population
#       Cross over using random members of the population?      To maintain diversity
#       Cutting out the bottom % of the population
#       And so on....
#   Possibly Later: Mutation
#   Iterations/Full-wrapper
