import numpy as np
import matplotlib.pyplot as plt
import os

def unpackLoadCondtions(loadConditions):
    forces = loadConditions['a']
    free = loadConditions['b']
    passive = loadConditions['c']
    formating_array = loadConditions['d']

    volfrac = formating_array[0]
    nelx = int(formating_array[1])
    nely = int(formating_array[2])
    penal = formating_array[3]
    rmin = formating_array[4]

    #print(volfrac,nelx,nely)

    return forces,free,passive,volfrac,nelx,nely,penal,rmin


def unpackIteration(iteration):
    x = iteration['a']
    xPhys = iteration['b']
    formating_array = iteration['c']

    compliance = formating_array[0]
    change = formating_array[1]
    mass = formating_array[2]

    #print(compliance,change,mass)
    return x,xPhys,compliance,change,mass

def cleanData(path):
    """
    The data set was generatied via the compliance minmization TopOpt, this method was prone to some numerical instabilities thus some data points are marked as invalid.
    This function will remove them.
    """

    AgentsToGrab = os.listdir(path)
    print("There are {} files to explore.".format(len(AgentsToGrab)))

    invalidAgents = []

    for agent in AgentsToGrab:
        agentFiles = os.listdir(os.path.join(path,agent))
        for fileName in agentFiles:
            if('Invalid_' in fileName):
                invalidAgents.append(agent)
                break
        
    print("Of the {} files scanned {} were invalid.".format(len(AgentsToGrab),len(invalidAgents)))

    for agent in invalidAgents:
        agentFiles = os.listdir(os.path.join(path,agent))
        for fileName in agentFiles:
            fileToRemove = os.path.join(path,agent,fileName)
            print("removing: {}".format(fileToRemove))
            os.remove(os.path.join(path,agent,fileName))
        directoryToRemove = os.path.join(path,agent)
        print("removing: {}\n".format(directoryToRemove))
        os.rmdir(os.path.join(path,agent))


        
        

def main():
    agentName = "Agent_432253"
    agentFileToGet = os.path.join(os.getcwd(),"Agents","30_30",agentName)

    FilesToGrab = os.listdir(agentFileToGet)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(agentFileToGet,fileName))
            print('loadCondtions Exist')
            
        elif('iteration_' in fileName):
            number_extension = fileName[len('iteration_'):]
            extesionIndex = number_extension.find('.')
            number = int(number_extension[:extesionIndex])
            #print(number)
            iterations.append([number,np.load(os.path.join(agentFileToGet,fileName))])
        #print(fileName)
    
    def sortKey(x):
        return x[0]

    iterations.sort(key=sortKey)
    #print(iterations)

    forces,free,passive,volfrac,nelx,nely,penal,rmin = unpackLoadCondtions(loadConditions)

    x_array = []
    xPhys_array = []
    compliance_array = []
    change_array = []
    mass_array = []

    for i in range(numberOfIterations):
        x,xPhys,compliance,change,mass = unpackIteration(iterations[i][1])
        x_array.append(x)
        xPhys_array.append(xPhys)
        compliance_array.append(compliance)
        change_array.append(change)
        mass_array.append(mass)


    





if(__name__ == "__main__"):
    AgentFolder = os.path.join(os.getcwd(),'MachineLerning','Data','100_50')
    cleanData(AgentFolder)  



