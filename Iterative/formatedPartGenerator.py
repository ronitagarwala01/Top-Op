import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tensorflow as tf

from premadePartGenerator.py import *


def main(formatVector):

    trueFormatVector = formatVector.copy()

    nelx = trueFormatVector[3]
    nely = trueFormatVector[4]
    model = getModel(nelx,nely)


    plotFormatVector(trueFormatVector,name='GeneratedPart/formatOut')
    #saveAsPVD(TruePart,100,50)
    shiftRadius = 2
    print("Predicting part")
    PredictedImages,shiftIndexes = iteratePartWithShift(model,trueFormatVector,numIterations=50,shiftAmnt=shiftRadius)
    
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

    truePart = np.reshape(TruePart,(101,51),order='F')
    scores = scoreModelPredictions(trueFormatVector,actualImages)
    sortedScoreIndexes = np.argsort(scores)
    for i in sortedScoreIndexes[:(shiftRadius+1)**2]:
        print("part {}: {:.3f} : ({},{})".format(i,scores[i],shiftIndexes[i][0],shiftIndexes[i][1]))

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
    SaveAsGif(bestImageIterations,100,50,"GeneratedPart/modelOutput")
    #import json
    #json.dump(saveStatsForEachIteration(bestImageIterations,trueFormatVector),open("modelStatsOverIteration.json",'w'))
    bestPart = np.reshape(actualImages[sortedScoreIndexes[0]],(101*51),order='F')
    # solution_list, objective_list, derivative_list, C_max, S_max, converged = convergenceTester(trueFormatVector,bestPart,0)
    # print(C_max)
    # print(S_max)
    # print(converged)

    part_flat = np.ravel(bestImageIterations[-1],order='F')

    #print("\tCompliance Max: {}".format(c_max))
    #print("\tStress Max: {}".format(s_max))

    
    f = open("GeneratedPart/ModelComparison.txt",'w')

    f.write("Circle 1: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(trueFormatVector[0][0][0],trueFormatVector[0][1][0],trueFormatVector[1][0]))
    f.write("Circle 2: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(trueFormatVector[0][0][1],trueFormatVector[0][1][1],trueFormatVector[1][1]))
    f.write("Circle 3: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(trueFormatVector[0][0][2],trueFormatVector[0][1][2],trueFormatVector[1][2]))

    f.write("\nForce 1: ( {:.2e}, {:.2e} ) magnitude = {:.3e}\n".format(trueFormatVector[2][0][0],trueFormatVector[2][1][0],np.sqrt(trueFormatVector[2][0][0]**2 + trueFormatVector[2][1][0]**2)))
    f.write("Force 2: ( {:.2e}, {:.2e} ) magnitude = {:.3e}\n".format(trueFormatVector[2][0][1],trueFormatVector[2][1][1],np.sqrt(trueFormatVector[2][0][1]**2 + trueFormatVector[2][1][1]**2)))
    f.write("Force 3: ( {:.2e}, {:.2e} ) magnitude = {:.3e}\n".format(trueFormatVector[2][0][2],trueFormatVector[2][1][2],np.sqrt(trueFormatVector[2][0][2]**2 + trueFormatVector[2][1][2]**2)))


    compliance,stress = convergenceTester(trueFormatVector,part_flat,1)
    mass = np.sum(part_flat)
    f.write("\nModel part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))
    f.close()

if(__name__ == "__main__"):
    #path = r'E:\TopoptGAfileSaves\Mass minimization\AlienWareData\True\100_50_Validation'
   
    formatVector = np.load('formatVector.npz')

    print(formatVector)

