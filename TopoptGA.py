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
    penaltyFActor = 2
    print("Evaluating Population of size {}:".format(len(population)))
    for i,member in enumerate(population):
        print("\t{:.2f}%".format(100*(i/len(population))),end ='\r')
        fitnessValue = fitnessFunction(member) + penaltyFActor*topOptFitnessFuntion(member,topOpt)
        memberFitnessValuePair.append(memberAndFitnessPairing(member, fitnessValue))
    print("\t100%          \nDone.")
    return memberFitnessValuePair

def topOptFitnessFuntion(member,topOpt:topOpter):
    """
    Takes a member of the population and scores the memeber based on it's compliance given the initial loads in the topOpt

    If the compliance is above the required ammount then add a large penatly
    Otherwise return 0
    """
    compliance,dc,K_unconstrained,K_constrained,u_unconstrained,u_constrained = topOpt.sensitivityAnalysis(member)

    if(False):
        fileSaver = AgentFileSaver(randint(0,9999),topOpt.nelx,topOpt.nely)

        
        fileSaver.saveCompressedFiles(  Compliance=np.array([compliance]),
                                        Compliance_Jacobian=dc,
                                        StiffnessMatrix_unconstrained=K_unconstrained.toarray(),
                                        StiffnessMatrix_constrained=K_constrained.toarray(),
                                        DisplacementVector_unconstrained=u_unconstrained,
                                        DisplacementVector_constrained=u_constrained,
                                        xPhys=member.astype("int32"))

    

    if(compliance > topOpt.complianceMax):
        return topOpt.nelx*topOpt.nely

    return 0

def applyConstraintsToPopulation(population,topOpt:topOpter):
    """
    Itterate through each memeber of the population and apply the problem constraints to each memeber.
    """

    for i in range(len(population)):
        population[i] = topOpt.applyConstraints(population[i])

def removeExessMaterial(member,matarialToRemove):
    return np.where(matarialToRemove==True,member,0)

def clearExessMaterialFromPopulation(population,materialToRemove,percentChanceToRemove:float = 1.0):
    for i in range(len(population)):
        if(np.random.random() <= percentChanceToRemove):
            population[i] = removeExessMaterial(population[i],materialToRemove)

def filterPopulationMutation(population,topOpt):
    newPopulation = []
    for member in population:
        newPopulation.append(topOpt.blurFilterCutoff(member,np.random.random()))
    return newPopulation

def testSave():
    nelx=10
    nely=10
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0
    maxCompliance = 10

    m = np.ones((nelx,nely))
    circle_1 = [.2,.3,.1,1,(3/2)*np.pi]
    circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
    circle_3 = [.8,.7,.1,1,(3/2)*np.pi]

    filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
    t = topOpter(nelx,nely,volfrac,penal,rmin,ft,maxCompliance)
    t.ApplyProblem(filledArea,supportArea,forceVector)

    print(topOptFitnessFuntion(m,t))

def testLoad():
    path = "Agents/10_10/Agent_3558/Agent3558.csv.npz"

    data = np.load(path,allow_pickle=True)

    print(data)
    arr0 = data['arr_0']
    print(arr0)
    for x in arr0:
        print(x)

def dispayMember(figure,figImage,member):
    if(plt.fignum_exists(figure.number)):
        figImage.set_array(member)
        figure.canvas.draw()
        figure.canvas.flush_events()

def main():
    #start by defining the topopt problem we will solve
    nelx=30
    nely=30
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0
    maxCompliance = 30

    DisplayFlag = True
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
    # and the counterclockwise angle of the force in degrees.
    circle_1 = [.15,.15,.1,1,(3/2)*np.pi]
    circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
    circle_3 = [.85,.85,.1,1,(3/2)*np.pi]

    filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
    t = topOpter(nelx,nely,volfrac,penal,rmin,ft,maxCompliance)
    t.ApplyProblem(filledArea,supportArea,forceVector)
    #t.applyCantileiverSetup()

    #import matplotlib to allow user to visualize the best agent of the current population
    if(DisplayFlag):
        plt.ion() # Ensure that redrawing is possible
        fig,ax = plt.subplots(1,1)
        im1 = ax.imshow(np.zeros((nelx,nely)), cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1))
        fig.show()


    numPop = 100
    numIterations = 100
    goalFitness = 10

    newPopulation = generateInitalPopulation(nelx, nely, numPop)
    clearExessMaterialFromPopulation(newPopulation,minViableArea,0.2)
    applyConstraintsToPopulation(newPopulation,t)


    if(DisplayFlag):
        dispayMember(fig,im1,newPopulation[-1])

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
        newFilteredMembers = filterPopulationMutation(newPopulation,t)

        for crossed in newCrossedMembers:
            newPopulation.append(crossed)

        for mutated in newMutatedMembers:
            newPopulation.append(mutated)
        
        for filtered in newFilteredMembers:
            newPopulation.append(filtered)

        applyConstraintsToPopulation(newPopulation,t)

        # Evaluation
        memberFitnessValuePairs = evaluation(newPopulation,t)
        print("Avg fitness: ", fitnessAverage(memberFitnessValuePairs, []))

        # Selection
        sortedPopulation = sortMemberFitnessValuePairs(memberFitnessValuePairs)

        if(DisplayFlag):
            #print(sortedPopulation[0])
            dispayMember(fig,im1,sortedPopulation[0][0])

        # Selection takes pop, numToSelect, and numElite
        # numToSelect is basically the population cap
        #print("select")
        selectedPopulation = selection(sortedPopulation, numPop, .2)

        if convergenceTest(selectedPopulation, goalFitness):
            print('"Converged"')
            print(selectedPopulation[0])

        newPopulation = extractSolutions(selectedPopulation)
    #input()


if __name__ == "__main__":
    main()