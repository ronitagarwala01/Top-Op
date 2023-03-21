"""Library dedicated to saving the intermediate data outputed by the fenics optimizer"""

import numpy as np
import os


"""
Optimized_array: list of 1D numpy arrays for each iteration of the optimizer
Derivatives_array: same as above but holds derivatives of the part.

formated = [Circles_Array, radii_array, forces_array
 nelx, nely, Youngs Modulus, ComplianceMax, StressMax]

 CirclesArray = [[c1_x, c2_x, c3_x], 
                 [c1_y, c2_y, c3_y]]

 Radii_array =  [c1_r,c2_r,c3_r]

 Forces_array = [[f1_x, f2_x, f3_x], 
                 [f1_y, f2_y, f3_y]]

 """

def loadDataset(folder):
    dataset = []
    conditions = []

    converged, nonConverged = 0, 0

    filesInFolder = os.listdir(folder)
    fileNames = [name for name in filesInFolder if name.startswith("Agent_")]

    for file in fileNames:
        try:
            unpacked, lastX, lastDer, lastObj= getData('Agents/' + str(file))
            converged += 1
        except:
            print(nonConverged)
            nonConverged += 1
            continue

        dataPoint = [lastX, lastDer, lastObj]

        dataset.append(dataPoint)
        conditions.append(unpacked)

    print("converged", converged)
    print("nonConverged", nonConverged)
    print("total", converged + nonConverged)

    return dataset, conditions
 

def getData(agentFileToGet):
    """
    Modified to return loadConditions and last iteration information

    Grabs and unpacks the data stored inside an agent file.
    To be used in conjunction with the fenics data creation and the saving format above.
    """
    FilesToGrab = os.listdir(agentFileToGet)
    iterations = []


    for fileName in FilesToGrab:

        if('loadConditions' in fileName):
            if fileName.startswith('loadConditions'):
                loadConditions = np.load(os.path.join(agentFileToGet,fileName))
            else:
                raise AttributeError
            
        elif('iteration_' in fileName):
            if fileName.startswith('iteration_'):
                numberStart = fileName.index('iteration_')
                number_extension = fileName[numberStart+len('iteration_'):]

                extesionIndex = number_extension.find('.')

                number = int(number_extension[:extesionIndex])

                iterations.append([number,np.load(os.path.join(agentFileToGet,fileName))])
            else:
                raise AttributeError

    
    def sortKey(x):
        return x[0]

    iterations.sort(key=sortKey)

    unpacked = unpackLoadConditions(loadConditions)

    lastIteration = iterations.pop(-1)
    x, obj, der = unpackIterations(lastIteration[1])
    
    return unpacked,x,der,obj

def unpackLoadConditions(loadConditions):
    """
    Modified, by Kyle, to meet the formatting needs of the cDCGAN model.
    
    Takes the dictionary created by np.load and unpacks the load conditions into an
    array shaped (20,), assuming that we are loading in 2d data.
    If it is 3d, it will return (21,) due to the inclusion of 'nelz'

    The contents are the same as the following, the only
    difference are that the sub-arrays are unpacked:

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

    # Formatting for cDCGAN model
    unpacked = []

    for x in range(3):
        unpacked.append(circles[0][x])
        unpacked.append(circles[1][x])
        unpacked.append(radii[x])
        unpacked.append(forces[0][x])
        unpacked.append(forces[1][x])

    unpacked.append(nelx)
    unpacked.append(nely)
    unpacked.append(Y)
    unpacked.append(C_max)
    unpacked.append(S_max)

    return np.array(unpacked)

def unpackIterations(iteration):
    """
    Modified to return information of the last iteration
    """

    x = iteration['a']
    derivative = iteration['b']
    obj = iteration['c']

    x = np.reshape(x, newshape=(51, 101))
    obj = np.reshape(obj, newshape=(51, 101))

    return x, derivative, obj



# dataset, conditions = loadDataset("Agents")

# print(dataset[0][0])
# print(dataset[0][1])
# print(dataset[0][2])

# print(conditions)