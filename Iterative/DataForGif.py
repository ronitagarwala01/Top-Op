from DemoNonNotebook import *
from problemStatementGenerator import *
from massopt_fenics import *

import numpy as np
import io
import os
import json

from PIL import Image
from time import perf_counter


def generateProblemOrientation(nelx, nely, C_max=2.0e-3, S_max=3.0e+7, Y=3.5e+11):

    xDim, yDim = calcRatio(nelx, nely) # Length, Width

    c1, c2, c3, forces = FenicsCircleAndForceGenerator(xDim, yDim)

    initial_conditions = [c1, c2, c3, forces, nelx, nely, Y, C_max, S_max]

    def unpackConditions(conditions):
        unpackedConditions = []

        circles = initial_conditions[:3]

        for x in range(3):
            for variable in circles[x]:
                unpackedConditions.append(variable)

            unpackedConditions.append(forces[0, x])
            unpackedConditions.append(forces[1, x])

        for x in range(4, len(conditions)):
            unpackedConditions.append(conditions[x])

        return unpackedConditions
        

    def formatForFenics(conditions):
        cx, cy = [], []
        formattedCircles, radii = [], []

        circles = initial_conditions[:3]

        for circle in circles:
            cx.append(circle[0])
            cy.append(circle[1])
            radii.append(circle[2])

        formattedCircles.append(np.array(cx))
        formattedCircles.append(np.array(cy))

        formattedCircles = np.array(formattedCircles)
        radii = np.array(radii)

        formattedConditions = [formattedCircles, radii]


        for x in range(3, len(conditions)):
            formattedConditions.append(conditions[x])

        return formattedConditions
    

    unpackedConditions = np.array(unpackConditions(initial_conditions))
    formattedConditions = np.array(formatForFenics(initial_conditions), dtype=object)

    return unpackedConditions, formattedConditions

def generateProblemConditions(formatted):
    # formatted = [circles, radii, forces, nelx, nely, Y, C_max, S_max]

    YoungsModulusMax = 5.0e+11
    YoungsModulusMin = 5.0e+10
    
    CmaxRatio = 50.0
    CminRatio = 2.0
    # ComplianceMinVal = 0

    SmaxRatio = 500.0
    SminRatio = 100.0
    # StressMinVal = 0

    y,c,s = createConstraints(YoungsModulusMin,YoungsModulusMax,CmaxRatio,CminRatio,SmaxRatio,SminRatio)

    formatted[5] = y
    formatted[6] = c
    formatted[7] = s

    return formatted
    
def generateData(nelx, nely, dontOptimize:bool=False):
    
    print("\n\n")
    _, formatted = generateProblemOrientation(nelx, nely)

    formatted = generateProblemConditions(formatted)

    print("Pass to fenics")
    start = perf_counter()
    if(not dontOptimize):
        solutions_list, objective_list, derivative_list, C_max, S_max, converged = fenicsOptimizer(formatted)
        formatted[6] = C_max
        formatted[7] = S_max
    else:
        solutions_list = [np.ones((nelx+1) * (nely+1))]
        converged = False
    end = perf_counter()

    
    return solutions_list, formatted, (end-start), converged

def iterateShiftDifferences(formated):

    nelx,nely = formated[3],formated[4]
    start = perf_counter()
    model = getModel(nelx,nely)
    #saveAsPVD(TruePart,100,50)
    shiftRadius = 2

    PredictedImages,shiftIndexes = iteratePartWithShift(model,formated,numIterations=50,shiftAmnt=shiftRadius)
    
    #print(shiftIndexes)

    actualImages = []
    subImages = PredictedImages[-1].shape[0]
    for i in range(subImages):
        shiftX = shiftIndexes[i][0]
        shiftY = shiftIndexes[i][1]

        #print("{}:({},{})".format(i,shiftX,shiftY))
        part = PredictedImages[-1][i,:,:,:]
        part = np.reshape(part,(nelx+1,nely+1))
        part = shiftImage(part,-shiftX,-shiftY)
        actualImages.append(part)

    
    scores = scoreModelPredictions(formated,actualImages)
    sortedScoreIndexes = np.argsort(scores)
    #for i in sortedScoreIndexes[:(shiftRadius+1)**2]:
    #    print("part {}: {:.3f} : ({},{})".format(i,scores[i],shiftIndexes[i][0],shiftIndexes[i][1]))

    bestImage = sortedScoreIndexes[0]
    bestImageIterations = [np.ones((nelx+1,nely+1))]
    for i in range(1,len(PredictedImages)):
        shiftX = shiftIndexes[bestImage][0]
        shiftY = shiftIndexes[bestImage][1]

        #print("{}:({},{})".format(i,shiftX,shiftY))
        part = PredictedImages[i][bestImage,:,:,:]
        part = np.reshape(part,(nelx+1,nely+1))
        part = shiftImage(part,-shiftX,-shiftY)
        bestImageIterations.append(part)
    #SaveAsGif(bestImageIterations,100,50,"modelOutput")

    #bestPart = np.reshape(actualImages[sortedScoreIndexes[0]],(101*51),order='F')
    # solution_list, objective_list, derivative_list, C_max, S_max, converged = convergenceTester(trueFormatVector,bestPart,0)
    # print(C_max)
    # print(S_max)
    # print(converged)
    end = perf_counter()

    return bestImageIterations,(end-start)

def plotFormatVector(formatVector,res:int=100,name:str='formatOut'):
    circles = formatVector[0]
    radii = formatVector[1]
    forces = formatVector[2]
    nelx, nely = formatVector[3], formatVector[4]
    Youngs, C_max, S_max = formatVector[5], formatVector[6], formatVector[7]
    print("Youngs:",Youngs)
    print("C_max:",C_max)
    print("S_max:",S_max)


    xDim,yDim = calcRatio(nelx,nely)
    x = np.linspace(0,xDim,res,True)
    y = np.linspace(0,yDim,res//2,True)

    X,Y = np.meshgrid(x,y)

    def dist(circleIndex):
        return np.sqrt((X-circles[0][circleIndex])**2 + (Y-circles[1][circleIndex])**2) - radii[circleIndex]
    
    circlesMap = np.minimum(dist(0),np.minimum(dist(1),dist(2)))
    circlesMap = np.where(circlesMap<=0,1,0)

    fig,ax = plt.subplots(1,1)
    plt.imshow(circlesMap,cmap='gray_r')
    MaxForce = np.max(np.abs(np.ravel(forces)))
    maxForceLength = res//10
    forceScale = maxForceLength/MaxForce

    def plotForce(num):
        centerX = circles[0][num] * res//2
        centerY = circles[1][num] * res//2
        dx = forces[0][num] * forceScale
        dy = forces[1][num] * forceScale
        ax.arrow(centerX,centerY,dx,dy,width=res/200,color='red')

    plotForce(0)
    plotForce(1)
    plotForce(2)
        
    
    plt.savefig(str(name) + ".png", format='png')


def createSaveFolder(formatVector,agentName:str = 'Sequence'):
    nelx, nely = formatVector[3], formatVector[4]

    #Check the current path and locate the correct folder to save the data to
    workingDirectory = os.getcwd()
    agentDirectory = os.path.join(workingDirectory,"output", 'ModelComparisons')
    dimesionFolder = os.path.join(agentDirectory,"{}_{}".format(nelx,nely))
    pathExists = os.path.exists(dimesionFolder)
    if( not pathExists):
        os.makedirs(dimesionFolder)

    #create a random agent number so that files can be seperated
    num = np.random.randint(1,999999)
    agentFolder = os.path.join(dimesionFolder,"{}_{}".format(agentName,num))
    pathExists = os.path.exists(agentFolder)
    if(not pathExists):
        os.makedirs(agentFolder)
    else:
        # if the agent folder currently exist then create an Agent#_ folder
        foundOpenNumber = False
        currentNumber = 1
        while( not foundOpenNumber):
            currentAgentN = os.path.join(dimesionFolder,"{}{}_{}".format(agentName,currentNumber,num))
            pathExists = os.path.exists(currentAgentN)
            if(pathExists):
                currentNumber += 1
            else:
                foundOpenNumber = True
        os.makedirs(currentAgentN)
        agentFolder = currentAgentN
    
    savedConditions = saveLoadConditions(agentFolder,formatVector)
    return agentFolder
    
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

def main(size:int=100,CreateFenicsPart:bool=True):
    nelx = size - (size%2)
    nely = nelx//2

    solutions_list, formatted, fenicsTime, converged = generateData(nelx,nely,not CreateFenicsPart)
    print(formatted)
    predictions_list,modelTime = iterateShiftDifferences(formatted)

    solution_asImages = []
    for i in range(len(solutions_list)):

        solution_asImages.append(np.reshape(solutions_list[i],(nelx+1,nely+1),order='F'))

    agentFolderPath = createSaveFolder(formatted)

    cwd = os.getcwd()
    os.chdir(agentFolderPath)
    plotFormatVector(formatted)
    if(CreateFenicsPart):
        SaveAsGif(solution_asImages,nelx,nely,"FenicsOutput")
    SaveAsGif(predictions_list,nelx,nely,"ModelOutput")
    #json.dump(saveStatsForEachIteration(predictions_list,formatted),open(os.path.join(agentFolderPath,"modelStatsOverIteration.json"),'w'))
    os.chdir(cwd)

    print("\n")
    print("Fenics took {:.2f} seconds.\nModel took {:.2f} seconds.\n".format(fenicsTime,modelTime))

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    
    part_flat = np.ravel(solution_asImages[-1],order='F')

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    compliance,stress = convergenceTester(formatted,part_flat,1)
    mass = np.sum(part_flat)
    f = open(os.path.join(agentFolderPath,"ModelComparison.txt"),'w')

    f.write("Circle 1: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][0],formatted[0][1][0],formatted[1][0]))
    f.write("Circle 2: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][1],formatted[0][1][1],formatted[1][1]))
    f.write("Circle 3: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][2],formatted[0][1][2],formatted[1][2]))

    f.write("\nForce 1: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][0],formatted[2][1][0],np.sqrt(formatted[2][0][0]**2 + formatted[2][1][0]**2)))
    f.write("Force 2: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][1],formatted[2][1][1],np.sqrt(formatted[2][0][1]**2 + formatted[2][1][1]**2)))
    f.write("Force 3: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][2],formatted[2][1][2],np.sqrt(formatted[2][0][2]**2 + formatted[2][1][2]**2)))

    f.write("\nYoung's Modulus: {:.5e}\n".format(formatted[5]))
    f.write("\nCompliance: {:.5f}\n".format(formatted[6]))
    f.write("\nStress: {:.5e}\n".format(formatted[7]))


    f.write("\nFenics part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    part_flat = np.ravel(predictions_list[-1],order='F')
    compliance,stress = convergenceTester(formatted,part_flat,1)
    mass = np.sum(part_flat)
    f.write("\nModel part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    f.write("\nFenics took {:.2f} seconds.\nModel took {:.2f} seconds.\n".format(fenicsTime,modelTime))
    if(converged and CreateFenicsPart):
        f.write("Fenics part converged.")
    else:
        f.write("Fenics part did not converge.")

    f.close()


def main2():
    nelx = 100
    nely = nelx//2 # 50
    _, formatedOriginal = generateProblemOrientation(nelx, nely)
    formatedOriginal = generateProblemConditions(formatedOriginal)

    formatedLR,formatedUD,formatedDiagonal = flipLoadConditions(formatedOriginal.copy())
    formatChangeForces,formatChangeValues,formatChangeCircles = changeFormatValues(formatedOriginal.copy())

    ModelPartDifferences(formatedOriginal,'Orginal')
    ModelPartDifferences(formatChangeForces,'Forces')
    ModelPartDifferences(formatChangeValues,'Values')
    ModelPartDifferences(formatChangeCircles,'Circles')
    ModelPartDifferences(formatedLR,'MirrorY')
    ModelPartDifferences(formatedUD,'MirrorX')
    ModelPartDifferences(formatedDiagonal,'MirrorDiagonal')

def main3():
    nelx = size - (size%2)
    nely = nelx//2

    solutions_list, formatted, fenicsTime, converged = generateData(nelx,nely,not CreateFenicsPart)
    print(formatted)
    predictions_list,modelTime = iterateShiftDifferences(formatted)

    solution_asImages = []
    for i in range(len(solutions_list)):

        solution_asImages.append(np.reshape(solutions_list[i],(nelx+1,nely+1),order='F'))

    agentFolderPath = createSaveFolder(formatted)

    cwd = os.getcwd()
    os.chdir(agentFolderPath)
    plotFormatVector(formatted)
    if(CreateFenicsPart):
        SaveAsGif(solution_asImages,nelx,nely,"FenicsOutput")
    SaveAsGif(predictions_list,nelx,nely,"ModelOutput")
    #json.dump(saveStatsForEachIteration(predictions_list,formatted),open(os.path.join(agentFolderPath,"modelStatsOverIteration.json"),'w'))
    os.chdir(cwd)

    print("\n")
    print("Fenics took {:.2f} seconds.\nModel took {:.2f} seconds.\n".format(fenicsTime,modelTime))

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    
    part_flat = np.ravel(solution_asImages[-1],order='F')

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    compliance,stress = convergenceTester(formatted,part_flat,1)
    mass = np.sum(part_flat)
    f = open(os.path.join(agentFolderPath,"ModelComparison.txt"),'w')

    f.write("Circle 1: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][0],formatted[0][1][0],formatted[1][0]))
    f.write("Circle 2: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][1],formatted[0][1][1],formatted[1][1]))
    f.write("Circle 3: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][2],formatted[0][1][2],formatted[1][2]))

    f.write("\nForce 1: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][0],formatted[2][1][0],np.sqrt(formatted[2][0][0]**2 + formatted[2][1][0]**2)))
    f.write("Force 2: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][1],formatted[2][1][1],np.sqrt(formatted[2][0][1]**2 + formatted[2][1][1]**2)))
    f.write("Force 3: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][2],formatted[2][1][2],np.sqrt(formatted[2][0][2]**2 + formatted[2][1][2]**2)))

    f.write("\nYoung's Modulus: {:.5e}\n".format(formatted[5]))
    f.write("\nCompliance: {:.5f}\n".format(formatted[6]))
    f.write("\nStress: {:.5e}\n".format(formatted[7]))


    f.write("\nFenics part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    part_flat = np.ravel(predictions_list[-1],order='F')
    compliance,stress = convergenceTester(formatted,part_flat,1)
    mass = np.sum(part_flat)
    f.write("\nModel part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    f.write("\nFenics took {:.2f} seconds.\nModel took {:.2f} seconds.\n".format(fenicsTime,modelTime))
    if(converged and CreateFenicsPart):
        f.write("Fenics part converged.")
    else:
        f.write("Fenics part did not converge.")

    f.close()  
    
def FenicsOptimizeTest(formatVector):
    
    print("\n\n")

    print("Pass to fenics")
    start = perf_counter()

    solutions_list, objective_list, derivative_list, C_max, S_max, converged = fenicsOptimizer(formatVector)
    formatVector[6] = C_max
    formatVector[7] = S_max
    
    end = perf_counter()

    
    return solutions_list, formatVector, (end-start), converged


def changeFormatValues(formatVector):
    circles = formatVector[0]
    radii = formatVector[1]
    forces = formatVector[2]
    nelx, nely = formatVector[3], formatVector[4]
    Youngs, C_max, S_max = formatVector[5], formatVector[6], formatVector[7]

    _, newFormat = generateProblemOrientation(nelx, nely)
    newFormat = generateProblemConditions(newFormat)

    formatChangeForces = [  circles.copy(),         radii.copy(),   newFormat[2].copy(),    nelx,nely,  Youngs, C_max,          S_max]
    formatChangeValues = [  circles.copy(),         radii.copy(),   forces.copy(),          nelx,nely,  Youngs, newFormat[6],   newFormat[7]]
    formatChangeCircles = [ newFormat[0].copy(),    radii.copy(),   forces.copy(),          nelx,nely,  Youngs, C_max,          S_max]

    return formatChangeForces,formatChangeValues,formatChangeCircles

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

def ModelPartDifferences(formatVector,trial:str='Sequence'):
    circles = formatVector[0]
    radii = formatVector[1]
    forces = formatVector[2]
    nelx, nely = formatVector[3], formatVector[4]
    Youngs, C_max, S_max = formatVector[5], formatVector[6], formatVector[7]

    solutions_list, formatted, fenicsTime, converged = FenicsOptimizeTest(formatVector)
    print(formatted)
    predictions_list,modelTime = iterateShiftDifferences(formatted)

    solution_asImages = []
    for i in range(len(solutions_list)):

        solution_asImages.append(np.reshape(solutions_list[i],(nelx+1,nely+1),order='F'))

    agentFolderPath = createSaveFolder(formatted,trial)

    cwd = os.getcwd()
    os.chdir(agentFolderPath)
    plotFormatVector(formatted)
    SaveAsGif(solution_asImages,nelx,nely,"FenicsOutput")
    SaveAsGif(predictions_list,nelx,nely,"ModelOutput")
    #json.dump(saveStatsForEachIteration(predictions_list,formatted),open(os.path.join(agentFolderPath,"modelStatsOverIteration.json"),'w'))
    os.chdir(cwd)

    print("\n")
    print("Fenics took {:.2f} seconds.\nModel took {:.2f} seconds.\n".format(fenicsTime,modelTime))

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    
    part_flat = np.ravel(solution_asImages[-1],order='F')

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    compliance,stress = convergenceTester(formatted,part_flat,1)
    mass = np.sum(part_flat)
    f = open(os.path.join(agentFolderPath,"ModelComparison.txt"),'w')

    f.write("Circle 1: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][0],formatted[0][1][0],formatted[1][0]))
    f.write("Circle 2: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][1],formatted[0][1][1],formatted[1][1]))
    f.write("Circle 3: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(formatted[0][0][2],formatted[0][1][2],formatted[1][2]))

    f.write("\nForce 1: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][0],formatted[2][1][0],np.sqrt(formatted[2][0][0]**2 + formatted[2][1][0]**2)))
    f.write("Force 2: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][1],formatted[2][1][1],np.sqrt(formatted[2][0][1]**2 + formatted[2][1][1]**2)))
    f.write("Force 3: ( {:.2e}, {:.2e} ) magnitued = {:.3e}\n".format(formatted[2][0][2],formatted[2][1][2],np.sqrt(formatted[2][0][2]**2 + formatted[2][1][2]**2)))

    f.write("\nYoung's Modulus: {:.5e}\n".format(formatted[5]))
    f.write("\nCompliance: {:.5f}\n".format(formatted[6]))
    f.write("\nStress: {:.5e}\n".format(formatted[7]))


    f.write("\nFenics part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    part_flat = np.ravel(predictions_list[-1],order='F')
    compliance,stress = convergenceTester(formatted,part_flat,1)
    mass = np.sum(part_flat)
    f.write("\nModel part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    f.write("\nFenics took {:.2f} seconds.\nModel took {:.2f} seconds.\n".format(fenicsTime,modelTime))
    if(converged):
        f.write("Fenics part converged.")
    else:
        f.write("Fenics part did not converge.")

    f.close()


if(__name__ == "__main__"):   
    main(CreateFenicsPart = False)    

