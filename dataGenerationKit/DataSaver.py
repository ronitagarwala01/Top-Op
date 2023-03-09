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



def saveData(formattedArray,iterationsArray,derivativesArray):
    """
    All in one save data function.
    Takes the formated arrays, the iteraions array ,and the derivatives array.
    Creates a new folder to save the data to and saves all the files as compressed numpy arrays.
    """

    nelx, nely = formattedArray[3], formattedArray[4]

    #Check the current path and locate the correct folder to save the data to
    workingDirectory = os.getcwd()
    agentDirectory = os.path.join(workingDirectory,"Agents")
    dimesionFolder = os.path.join(agentDirectory,"{}_{}".format(nelx,nely))
    pathExists = os.path.exists(dimesionFolder)
    if( not pathExists):
        os.makedirs(dimesionFolder)

    #create a random agent number so that files can be seperated
    num = np.random.randint(1,999999)
    agentFolder = os.path.join(dimesionFolder,"Agent_{}".format(num))
    pathExists = os.path.exists(agentFolder)
    if(not pathExists):
        os.makedirs(agentFolder)
    else:
        # if the agent folder currently exist then create an Agent#_ folder
        foundOpenNumber = False
        currentNumber = 1
        while( not foundOpenNumber):
            currentAgentN = os.path.join(dimesionFolder,"Agent{}_{}".format(currentNumber,num))
            pathExists = os.path.exists(currentAgentN)
            if(pathExists):
                currentNumber += 1
            else:
                foundOpenNumber = True
        os.makedirs(currentAgentN)
        agentFolder = currentAgentN
    

    #using the created folder save the initial load conditions
    savedConditions = saveLoadConditions(agentFolder,formattedArray)
    
    #save the iterations arrays
    savedIterations = saveIteration(agentFolder,iterationsArray,derivativesArray)

    #if there was an error(either bool is false) in saving the load conditions or the iterations, mark the folder as invalid
    if((savedConditions and savedIterations) == False):
        markAsInvalid(agentFolder)
    
def saveLoadConditions(folderToSaveTo,formattedArray):
    """
    Saves the load conditions stored inside the formatted Array

    returns true if data was saved, returns false if there was an error
    """

    #unpack data
    circles = formattedArray[0]
    radii = formattedArray[1]
    forces = formattedArray[2]
    nelx, nely = formattedArray[3], formattedArray[4]
    Y, C_max, S_max = formattedArray[5], formattedArray[6], formattedArray[7]

    #format the directory to save the data to
    originalWorkingDirectory = os.getcwd()
    os.chdir(folderToSaveTo)
    fileNameToSaveAs = "loadConditions"
    formating_array = np.array([nelx,nely,Y,C_max,S_max])
    
    #try to save the data
    dataIsSaved = False
    try:
        np.savez_compressed(fileNameToSaveAs,a=circles,b=radii,c=forces,d=formating_array)
        dataIsSaved = True
    except:
        print("Something went wrong.")
        print("Tried to save: {}".format(fileNameToSaveAs))
    os.chdir(originalWorkingDirectory)
    return dataIsSaved

def saveIteration(folderToSaveTo,iterationsArray,derivativesArray):
    """
    Saves the individule interatins for X and the derivative of X

    returns true if data was saved, returns false if there was an error
    """
    originalWorkingDirectory = os.getcwd()
    os.chdir(folderToSaveTo)
    numFailed = 0
    for i in range(len(iterationsArray)):
        fileNameToSaveAs = f"iteration_{i}"
        
        #unpack data
        ar1 = iterationsArray[i]
        ar2 = derivativesArray[i]
        
        try:
            np.savez_compressed(fileNameToSaveAs,a=ar1,b=ar2)
        except:
            print("Something went wrong.")
            print("Tried to save: {}".format(fileNameToSaveAs))
            numFailed += 1
    os.chdir(originalWorkingDirectory)


    #if no data was saved return false
    if (numFailed == len(iterationsArray)):
        return False
    return True

def markAsInvalid(folderToSaveTo):
    #Mark that the solution is invalid thus should not be used
    originalWorkingDirectory = os.getcwd()
    os.chdir(folderToSaveTo)

    filesInDirectory = os.listdir()

    for file in filesInDirectory:
        os.rename(file,str("Invalid_" + file))



    os.chdir(originalWorkingDirectory)


def getData(agentFileToGet):
    """
    Grabs and unpacks the data stored inside an agent file.
    To be used in conjunction with the fenics data creation and the saving format above.
    """

    FilesToGrab = os.listdir(agentFileToGet)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(agentFileToGet,fileName))
            #print('loadCondtions Exist')
            
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

    formated = unpackLoadConditions(loadConditions)
    x_array = []
    derivatives_array = []
    
    for num,arrays in iterations:
        x,der = unpackIterations(arrays)
        x_array.append(x)
        derivatives_array.append(der)
    
    return formated,x_array,derivatives_array

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

    return x,derivative
