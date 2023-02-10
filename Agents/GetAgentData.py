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

    #print(volfrac,nelx,nely)

    return forces,free,passive,volfrac,nelx,nely


def unpackIteration(iteration):
    x = iteration['a']
    xPhys = iteration['b']
    formating_array = iteration['c']

    compliance = formating_array[0]
    change = formating_array[1]
    mass = formating_array[2]

    #print(compliance,change,mass)
    return x,xPhys,compliance,change,mass



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

    forces,free,passive,volfrac,nelx,nely = unpackLoadCondtions(loadConditions)

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
    main()  