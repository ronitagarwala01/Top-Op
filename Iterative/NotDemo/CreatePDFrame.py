
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from Demo_test import *
from fenics_tester import *



COLUMNS = ["nelx","nely","did converge","Youngs modulus",
            "circle 1 x","circle 1 y","circle 1 radius","force 1 x","force 1 y",
            "circle 2 x","circle 2 y","circle 2 radius","force 1 x","force 2 y",
            "circle 3 x","circle 3 y","circle 3 radius","force 3 x","force 3 y",
            "True Mass","True Compliance","True Stress",
            "Predicted Mass","Predicted Stress","Predicted Compliance"]



def getPartStats(formatVector,parts_array):
    print("Testing Parts")
    mass_array = []
    comp_array = []
    stress_array = []

    for i in range(len(parts_array)):
        part_flat = np.ravel(parts_array[i],order='F')
        #print("\tCompliance Max: {}".format(c_max))
        #print("\tStress Max: {}".format(s_max))

        compliance,stress = convergenceTester(formatVector,part_flat)

        mass_array.append(np.sum(part_flat))
        comp_array.append(compliance)
        stress_array.append(stress)
    
    return mass_array,comp_array,stress_array
        

def getBestImage(predictedImages,shiftIndexes,formatVector):

    actualImages = []
    subImages = predictedImages[-1].shape[0]
    for i in range(subImages):
        shiftX = shiftIndexes[i][0]
        shiftY = shiftIndexes[i][1]

        #print("{}:({},{})".format(i,shiftX,shiftY))
        part = predictedImages[-1][i,:,:,:]
        part = np.reshape(part,(101,51))
        part = shiftImage(part,-shiftX,-shiftY)
        actualImages.append(part)

    C_max, S_max = formatVector[6], formatVector[7]
    mass_array,compliance_array,stress_array = getPartStats(formatVector,actualImages)

    #the best part is the part with no stress of compliance violation with the least mass
    # if all parts break the constraints, then return the part with the best score
    scores = []
    possibleParts = []
    for i in range(len(actualImages)):
        if(compliance_array[i] <= C_max and stress_array[i] <=S_max):
            possibleParts.append(i)
        score = mass_array[i] + np.exp(compliance_array[i]/C_max) + np.exp(stress_array[i]/S_max) - np.exp(2)
        scores.append(score)
    
    if(len(possibleParts) >= 1):
        minMass = np.inf
        index = 0
        for i in possibleParts:
            if(mass_array[i] < minMass):
                index = i
                minMass = mass_array[i]
        
        return mass_array[index],compliance_array[index],stress_array[index]
    
    else:
        minScore = np.argsort(scores)
        index = minScore[0]

        return mass_array[index],compliance_array[index],stress_array[index]


def buildFrame(path):
    model = getModel(100,50)
    frameData = []
    pointsToGrab = os.listdir(path)

    for i in range(151,len(pointsToGrab)):
        fileName = pointsToGrab[i]

        print("{}:{:.2f}%\t".format(i,100*(i/len(pointsToGrab))),end='\r')

        dataPath = os.path.join(path,fileName)
        formatVector,part,converged = loadFenicPart(dataPath)

        circles = formatVector[0]
        radii = formatVector[1]
        forces = formatVector[2]
        nelx, nely = formatVector[3], formatVector[4]
        Y, C_max, S_max = formatVector[5], formatVector[6], formatVector[7]
        C_max,S_max = convergenceTester(formatVector,part)

        PredictedImages,shiftIndexes = iteratePartWithShift(model,formatVector.copy(),shiftAmnt=2)

        predMass,predCompliance,predStress = getBestImage(PredictedImages,shiftIndexes,formatVector)
        mass = np.sum(part)

        print("\t{:.1f}:{:.4f}:{:.2f}\n\t{:.1f}:{:.4f}:{:.2f}".format(mass,C_max,S_max,predMass,predCompliance,predStress))

        newFrame = [nelx, nely,converged,Y,
                    circles[0][0],circles[1][0],radii[0],forces[0][0],forces[1][0],
                    circles[0][1],circles[1][1],radii[1],forces[0][1],forces[1][1],
                    circles[0][2],circles[1][2],radii[2],forces[0][2],forces[1][2],
                    mass,C_max,S_max,
                    predMass,predCompliance,predStress]
        
        print(newFrame)

        if(i%10 == 0 and i > 1):
            print("saving Temp data")

            df = pd.DataFrame(frameData,columns=COLUMNS)
            df.to_csv('data{}.csv'.format(i//10))

        frameData.append(newFrame)
    print("100%     ")

    df = pd.DataFrame(frameData,columns=COLUMNS)
    return df

if(__name__ == "__main__"):
    path =os.path.join(os.getcwd(),'Data','100_50_Validation')
    df = buildFrame(path)
    print(df)

    fileName = 'data1.csv'
    df.to_csv(fileName)
