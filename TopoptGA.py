import numpy as np

from GA import *
from GA_score_compliance import topOpter
from ProblemMapper import *
from FileSaver import AgentFileSaver
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider
from random import randint

"""
Wrapped within an iterator:
    Evaluation
        For each member within the population:
            Pass member to fitness function:
                Fitness function (element minimization):
                    Sum the value of each array
                    Return sum
            Pair each member with fitness value
"""
def memberAndFitnessPairing(member, fitnessValue):
    memberFitnessTuple = (member, fitnessValue)

    return memberFitnessTuple

def fitnessFunction(member):
    fitnessValue = np.sum(member)

    return fitnessValue

def evaluation(population,topOpt:topOpter):
    memberFitnessValuePair = []

    for member in population:
        fitnessValue = fitnessFunction(member) + topOptFitnessFuntion(member,topOpt)
        memberFitnessValuePair.append(memberAndFitnessPairing(member, fitnessValue))

    return memberFitnessValuePair

def topOptFitnessFuntion(member,topOpt:topOpter):
    """
    Takes a member of the population and scores the memeber based on it's compliance given the initial loads in the topOpt

    If the compliance is above the required ammount then add a large penatly
    Otherwise return 0
    """
    compliance,dc,K_unconstrained,K_constrained,u_unconstrained,u_constrained = topOpt.sensitivityAnalysis(member)
    fileSaver = AgentFileSaver(randint(0,9999),topOpt.nelx,topOpt.nely)

    fileSaver.saveNumpyArray("Compliance",np.array([compliance]))
    fileSaver.saveNumpyArray("Compliance_Jacobian",dc)
    fileSaver.saveNumpyArray("StiffnessMatrix_unconstrained",K_unconstrained.toarray())
    fileSaver.saveNumpyArray("StiffnessMatrix_constrained",K_constrained.toarray())
    fileSaver.saveNumpyArray("DisplacementVector_unconstrained",u_unconstrained)
    fileSaver.saveNumpyArray("DisplacementVector_constrained",u_constrained)
    fileSaver.saveNumpyArray("xPhys",member.astype("int32"))

    

    if(compliance > topOpt.complianceMax):
        return topOpt.nelx*topOpt.nely

    return 0

def applyConstraintsToPopulation(population,topOpt:topOpter):
    """
    Itterate through each memeber of the population and apply the problem constraints to each memeber.
    """

    for i in range(len(population)):
        population[i] = topOpt.applyConstraints(population[i])

if __name__ == "__main__":
    #start by defining the topopt problem we will solve
    nelx=10
    nely=10
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0
    maxCompliance = 10
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
    # and the counterclockwise angle of the force in degrees.
    circle_1 = [.2,.3,.1,1,(3/2)*np.pi]
    circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
    circle_3 = [.8,.7,.1,1,(3/2)*np.pi]

    filledArea,supportArea,forceVector = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
    t = topOpter(nelx,nely,volfrac,penal,rmin,ft,maxCompliance)
    t.ApplyProblem(filledArea,supportArea,forceVector)

    #import matplotlib to allow user to visualize the best agent of the current population
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots(1,1)
    im1 = ax.imshow(np.zeros((nelx,nely)), cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1))
    fig.show()


    numPop = 100
    numIterations = 100
    goalFitness = 10

    newPopulation = generateInitalPopulation(nelx, nely, numPop)
    applyConstraintsToPopulation(newPopulation,t)

    for x in range(numIterations):
        print("Iteration:", x)

        # Duplication
        toCross = np.copy(newPopulation)
        toMutate = np.copy(newPopulation)

        # This will be passed raw to the crossover algorithm
        np.random.shuffle(toCross)

        # Crossover & Mutation
        newCrossedMembers = crossover(toCross)
        newMutatedMembers = mutation(toMutate)

        for crossed in newCrossedMembers:
            newPopulation.append(crossed)

        for mutated in newMutatedMembers:
            newPopulation.append(mutated)

        applyConstraintsToPopulation(newPopulation,t)

        # Evaluation
        memberFitnessValuePairs = evaluation(newPopulation,t)
        print("Avg fitness: ", fitnessAverage(memberFitnessValuePairs, []))

        # Selection
        sortedPopulation = sortMemberFitnessValuePairs(memberFitnessValuePairs)

        if(plt.fignum_exists(fig.number)):
            #print(sortedPopulation[0])
            im1.set_array(sortedPopulation[0][0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Selection takes pop, numToSelect, and numElite
        # numToSelect is basically the population cap
        #print("select")
        selectedPopulation = selection(sortedPopulation, .5, .2)

        if convergenceTest(selectedPopulation, goalFitness):
            print('"Converged"')
            print(selectedPopulation[0])

        newPopulation = extractSolutions(selectedPopulation)