import numpy as np

"""
List of functions that need to be written (18 functions):
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

    dtype = [('member', object), ('fitnessValue', int)]

    return np.array(memberFitnessTuple, dtype=dtype)


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