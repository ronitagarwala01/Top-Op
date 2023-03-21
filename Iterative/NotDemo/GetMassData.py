import numpy as np
import os


def getData(agentFileToGet):
    """
    Grabs and unpacks the data stored inside an agent file.
    To be used in conjunction with the fenics data creation and the saving format above.
    """
    #print(agentFileToGet)
    FilesToGrab = os.listdir(agentFileToGet)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []
    markName = ''


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(agentFileToGet,fileName))
            #print('loadCondtions Exist')
            markIndex = fileName.index('loadConditions')
            if(markIndex > 0):
                markName = fileName[:markIndex-1]
            
        elif('iteration_' in fileName):
            numberStart = fileName.index('iteration_')
            number_extension = fileName[numberStart+len('iteration_'):]
            extesionIndex = number_extension.find('.')
            number = int(number_extension[:extesionIndex])
            #print(number)
            iterations.append([number,np.load(os.path.join(agentFileToGet,fileName))])
        #print(fileName)
    
    def sortKey(x):
        return x[0]

    iterations.sort(key=sortKey)

    formated = unpackLoadConditions(loadConditions)
    x_array = []
    derivatives_array = []
    objectives_array = []
    
    for num,arrays in iterations:
        x,obj,der = unpackIterations(arrays)
        x_array.append(x)
        derivatives_array.append(der)
        objectives_array.append(obj)
    
    return formated,x_array,derivatives_array,objectives_array,markName

def unpackLoadConditions(loadConditions):
    """
    Takes the dictionary created by np.load and unpacks the load conditions in the format:

    formated = [Circles_Array, radii_array, forces_array
                nelx, nely, Youngs Modulus, ComplianceMax, StressMax]
    """
    circles = loadConditions['a']
    radii = loadConditions['b']
    forces = loadConditions['c']
    formating_array = loadConditions['d']

    nelx = int(formating_array[0])
    nely = int(formating_array[1])
    Y = formating_array[2]
    C_max = formating_array[3]
    S_max = formating_array[4]

    formated = [circles,radii,forces,nelx,nely,Y,C_max,S_max]
    return formated

def unpackIterations(iteration):
    x = iteration['a']
    derivative = iteration['b']
    obj = iteration['c']

    return x,derivative,obj
