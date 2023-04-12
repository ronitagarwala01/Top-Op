from DemoNonNotebook import *
from problemStatementGenerator import *
from massopt_fenics import *

import numpy as np
import io

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
    
def generateData(nelx, nely):
    
    print("\n\n")
    _, formatted = generateProblemOrientation(nelx, nely)

    formatted = generateProblemConditions(formatted)

    print("Pass to fenics")
    start = perf_counter()
    solutions_list, objective_list, derivative_list, C_max, S_max, converged = fenicsOptimizer(formatted)
    end = perf_counter()

    formatted[6] = C_max
    formatted[7] = S_max
    
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
        part = np.reshape(part,(101,51))
        part = shiftImage(part,-shiftX,-shiftY)
        actualImages.append(part)

    
    scores = scoreModelPredictions(formated,actualImages)
    sortedScoreIndexes = np.argsort(scores)
    #for i in sortedScoreIndexes[:(shiftRadius+1)**2]:
    #    print("part {}: {:.3f} : ({},{})".format(i,scores[i],shiftIndexes[i][0],shiftIndexes[i][1]))

    bestImage = sortedScoreIndexes[0]
    bestImageIterations = [np.ones((101,51))]
    for i in range(1,len(PredictedImages)):
        shiftX = shiftIndexes[bestImage][0]
        shiftY = shiftIndexes[bestImage][1]

        #print("{}:({},{})".format(i,shiftX,shiftY))
        part = PredictedImages[i][bestImage,:,:,:]
        part = np.reshape(part,(101,51))
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
        endX = centerX - forces[0][num] * forceScale
        endY = centerY - forces[1][num] * forceScale
        x1 = [centerX,endX]
        y1 = [centerY,endY]
        ax.plot(x1,y1)

    plotForce(0)
    plotForce(1)
    plotForce(2)
        
    
    plt.savefig(str(name) + ".png", format='png')

if(__name__ == "__main__"):
    nelx = 100
    nely = nelx//2
    solutions_list, formatted, fenicsTime, converged = generateData(nelx,nely)
    predisctions_list,modelTime = iterateShiftDifferences(formatted)

    solution_asImages = []
    for i in range(len(solutions_list)):

        solution_asImages.append(np.reshape(solutions_list[i],(nelx+1,nely+1),order='F'))


    plotFormatVector(formatted)
    SaveAsGif(solution_asImages,nelx,nely,"FenicsOutput")
    SaveAsGif(predisctions_list,nelx,nely,"ModelOutput")

    print("\n")
    print("Fenics took {:.2f} seconds.\nModel took {:.2f} seconds.".format(fenicsTime,modelTime))

