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



def saveData(formattedArray,iterationsArray, objectivesArray, derivativesArray,converged):
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
    savedIterations = saveIteration(agentFolder,iterationsArray,objectivesArray,derivativesArray)

    #if there was an error(either bool is false) in saving the load conditions or the iterations, mark the folder as invalid
    print("Agent {} saved to path:\n\t{}".format(num,agentFolder))
    if(((savedConditions and savedIterations) == False)):
        markAs(agentFolder,"Invalid")
        print("There was an error saving the data. Marked as Invalid.")
    elif(converged == False):
        markAs(agentFolder,"NotConverged")
        print("Data did not converge. Marked as NotConverged.")
    
    
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

def saveIteration(folderToSaveTo,iterationsArray,objectivesArray,derivativesArray):
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
        ar2 = objectivesArray[i]
        ar3 = derivativesArray[i]
        
        try:
            np.savez_compressed(fileNameToSaveAs,a=ar1,b=ar2,c=ar3)
        except:
            print("Something went wrong with saving iteraion: {}".format(i))
            print("Tried to save: {}".format(fileNameToSaveAs))
            numFailed += 1
    os.chdir(originalWorkingDirectory)


    #if no data was saved return false
    if (numFailed == len(iterationsArray)):
        return False
    return True

def markAs(folderToSaveTo,markString:str = "Invalid"):
    #Mark that the solution is invalid thus should not be used
    originalWorkingDirectory = os.getcwd()
    os.chdir(folderToSaveTo)

    filesInDirectory = os.listdir()

    for file in filesInDirectory:
        os.rename(file,str(markString + "_" + file))



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
    objectives_array = []
    
    for num,arrays in iterations:
        x,der,obj = unpackIterations(arrays)
        x_array.append(x)
        derivatives_array.append(der)
        objectives_array.append(obj)
    
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
    obj = iteration['c']

    return x,derivative,obj


def flipLoadConditions(formattedArray):
    """
    Takes the formatted Array and returns three variations with the circles fliped horizontally, vertically, and over the diagonal.

    Parameters:
        - formattedArray
    
    returns:
        - formatted LR: fliped left right(horizontal mirror)
        - formatted UD: fliped up down(vertical mirror)
        - formatted diagnal: fliped over both horizontal and vertical
    """
    circles = formattedArray[0]
    radii = formattedArray[1]
    forces = formattedArray[2]
    nelx, nely = formattedArray[3], formattedArray[4]
    Y, C_max, S_max = formattedArray[5], formattedArray[6], formattedArray[7]

    #create arrays for circle positions
    circlesLR = circles.copy()
    circlesUD = circles.copy()
    circlesDiagonal = circles.copy()

    #flip the x positions
    circlesLR[0,:] = 2-circlesLR[0,:]

    #flip y postions
    circlesUD[1,:] = 1-circlesUD[1,:]

    #flip diagonal
    circlesDiagonal[0,:] = circlesLR[0,:]
    circlesDiagonal[1,:] = circlesUD[1,:]

    forcesLR = forces.copy()
    forcesUD = forces.copy()
    forcesDiagonal = forces.copy()

    #flip the x magnitude
    forcesLR[0,:] = -forcesLR[0,:]

    #flip y magnitude
    forcesUD[1,:] = -forcesUD[1,:]

    #flip diagonal
    forcesDiagonal[0,:] = forcesLR[0,:]
    forcesDiagonal[1,:] = forcesUD[1,:]

    #duplicate
    formattedLR = [circlesLR,radii,forcesLR,nelx,nely,Y,C_max,S_max]
    formattedUD = [circlesUD,radii,forcesUD,nelx,nely,Y,C_max,S_max]
    formattedDiagonal = [circlesDiagonal,radii,forcesDiagonal,nelx,nely,Y,C_max,S_max]

    return formattedLR,formattedUD,formattedDiagonal


    



