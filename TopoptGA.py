import numpy as np

from GA import *
from GA_score_compliance import topOpter
from ProblemMapper import *
from FileSaver import AgentFileSaver
import matplotlib.pyplot as plt
from matplotlib import colors
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
        fitnessValue = fitnessFunction(member) + penaltyFActor*topOptFitnessFuntion(member,topOpt,np.random.random() < .01)
        memberFitnessValuePair.append(memberAndFitnessPairing(member, fitnessValue))
    print("\t100%          \nDone.")
    return memberFitnessValuePair

def topOptFitnessFuntion(member,topOpt:topOpter, saveFlag:bool = False):
    """
    Takes a member of the population and scores the memeber based on it's compliance given the initial loads in the topOpt

    If the compliance is above the required ammount then add a large penatly
    Otherwise return 0
    """
    compliance,dc,K_unconstrained,K_constrained,u_unconstrained,u_constrained = topOpt.sensitivityAnalysis(member)

    if(saveFlag):
        fileSaver = AgentFileSaver(randint(0,99999),topOpt.nelx,topOpt.nely)
        fileSaver.save_GA_data_compressed(member.astype("int32"),topOpt.f,topOpt.free,compliance,topOpt.complianceMax)

    

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

        newmemeber = topOpt.blurFilterCutoff(member,np.random.random())
        child = CrossoverBooleanAnd([member,newmemeber])
        newPopulation.append(child)
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

    print(topOptFitnessFuntion(m,t,saveFlag=True))




def testLoad():
    agentNUm = 5132
    path = "Agents/10_10/Agent_{}/Agent{}.csv.npz".format(agentNUm,agentNUm)

    data = np.load(path,allow_pickle=True)

    print(data)
    print(data['a'])
    print(data['b'])
    print(data['c'])
    print(data['d'])

def testMutate():
    ar1 = np.random.choice([0,1],size=5)
    ar2 = np.random.choice([0,1],size=5)

    print(ar1)
    print(ar2)

    print(CrossoverBooleanAnd([ar1,ar2]))



def dispayMember(figure,figImage,member):
    if(plt.fignum_exists(figure.number)):
        figImage.set_array(member)
        figure.canvas.draw()
        figure.canvas.flush_events()

def randomCircleGenerator():
    x1 = np.random.random()
    y1 = np.random.random()
    r1 = np.random.random()/3
    f1 = np.random.random()*3
    a1 = np.random.random()*2*np.pi
    c1 = [x1,y1,r1,f1,a1]
    return c1

def main(maxCompliance):
    #start by defining the topopt problem we will solve
    nelx=30
    nely=30
    volfrac=0.4
    rmin=1.5
    penal=3.0
    ft=0

    DisplayFlag = False
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
    # and the counterclockwise angle of the force in degrees.
    
    circle_1 = [.15,.15,.1,1,(3/2)*np.pi]
    circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
    circle_3 = [.85,.85,.1,1,(3/2)*np.pi]
    setupTries = 10
    canSetUp = False
    for i in range(setupTries):
        try:
            circle_1 = randomCircleGenerator()
            circle_2 = randomCircleGenerator()
            circle_3 = randomCircleGenerator()
        except:
            canSetUp = False
        else:
            canSetUp = True
            break
    
    if(canSetUp == False):
        return

    filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
    t = topOpter(nelx,nely,volfrac,penal,rmin,ft,maxCompliance)
    t.ApplyProblem(filledArea,supportArea,forceVector)
    #t.applyCantileiverSetup()

    solidMember = removeExessMaterial(np.ones((nelx,nely)),minViableArea)
    solidMember = t.applyConstraints(solidMember)

    plt_x_subplots = 2
    plt_y_subplots = 2
    im_array = []
    #import matplotlib to allow user to visualize the best agent of the current population
    if(DisplayFlag):
        plt.ion() # Ensure that redrawing is possible
        fig,ax = plt.subplots(plt_x_subplots,plt_y_subplots)
        for x1 in range(plt_x_subplots):
            for y1 in range(plt_y_subplots):
                im_array.append(ax[x1,y1].imshow(np.zeros((nelx,nely)), cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1)))
        fig.show()


    numPop = 100
    numIterations = 50

    newPopulation = generateInitalPopulation(nelx, nely, numPop)
    clearExessMaterialFromPopulation(newPopulation,minViableArea,0.2)
    applyConstraintsToPopulation(newPopulation,t)


    if(DisplayFlag):
        dispayMember(fig,im_array[0],newPopulation[-1])

    for x in range(numIterations):
        print("Iteration:", x)

        # Duplication
        toCross = np.copy(newPopulation)
        toMutate = np.copy(newPopulation)

        # This will be passed raw to the crossover algorithm
        np.random.shuffle(toCross)

        # Crossover & Mutation
        newCrossedMembers = crossover(toCross,4)
        newMutatedMembers = mutation(toMutate)
        newFilteredMembers = filterPopulationMutation(newPopulation,t)

        for crossed in newCrossedMembers:
            newPopulation.append(crossed)

        for mutated in newMutatedMembers:
            newPopulation.append(mutated)
        
        for filtered in newFilteredMembers:
            newPopulation.append(filtered)

        if(np.random.random() < 0.1):
            newPopulation.append(solidMember.copy())

        applyConstraintsToPopulation(newPopulation,t)

        # Evaluation
        memberFitnessValuePairs = evaluation(newPopulation,t)
        print("Avg fitness: ", fitnessAverage(memberFitnessValuePairs, []))

        # Selection
        sortedPopulation = sortMemberFitnessValuePairs(memberFitnessValuePairs)

        if(DisplayFlag):
            #print(sortedPopulation[0])
            for x1 in range(plt_x_subplots):
                for y1 in range(plt_y_subplots):
                    dispayMember(fig,im_array[x1*plt_x_subplots + y1],sortedPopulation[x1*plt_x_subplots + y1][0])

        # Selection takes pop, numToSelect, and numElite
        # numToSelect is basically the population cap
        #print("select")
        selectedPopulation = selection(sortedPopulation, numPop, .2)
        newPopulation = extractSolutions(selectedPopulation)
    #input()


if __name__ == "__main__":
    for i in range(100):
        main(randint(10,40))