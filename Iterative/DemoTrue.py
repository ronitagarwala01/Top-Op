import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from DemoSuportLibrary import *

import os
from time import time

import tensorflow as tf

FORCE_NORMILIZATION_FACTOR = 7000
YOUNGS_MODULUS_NORMILIZATION_FACTOR = 238000000000
COMPLIANCE_MAX_NORMILIZATION_FACTOR = 0.03
STRESS_MAX_NORMILIZATION_FACTOR = 15000000



#code to create true force vectors
def genForces(f1,a1,f2,a2):

    fx_1 = f1*np.cos(a1)
    fy_1 = f1*np.sin(a1)
    fx_2 = f2*np.cos(a2)
    fy_2 = f2*np.sin(a2)

    fx_3 = -(fx_1+fx_2)
    fy_3 = -(fy_1+fy_2)

    print("forces are:")
    print("Force1: x={}, y={}".format(int(fx_1),int(fy_1)))
    print("Force2: x={}, y={}".format(int(fx_2),int(fy_2)))
    print("Force3: x={}, y={}".format(int(fx_3),int(fy_3)))

    return np.array([[fx_1,fx_2,fx_3],[fy_1,fy_2,fy_3]])

def getModel(nelx,nely):
    modelNum = 9
    #model_half_Resolution = Model_m9(nelx//2+1,nely//2+1)
    model_Full_Resolution = Model_m9(nelx+1,nely+1)
    fileSaveName = "Model_m{}".format(modelNum)
    
    

    modelPath = os.path.join(os.getcwd(),'ModelSave',fileSaveName)
    
    if(os.path.isdir(modelPath)):
        try:
            
            #model_half_Resolution.load_weights(os.path.join(modelPath,fileSaveName))
            model_Full_Resolution.load_weights(os.path.join(modelPath,fileSaveName))
        except:
            print("Model weights could not be loaded.")
        else:
            print("Model weights Loaded")
    else:
        print("Model file does not exist.")

    #return model_half_Resolution,model_Full_Resolution\
    return model_Full_Resolution

def formatDataForModel(formatVector):
    circles = formatVector[0]
    radii = formatVector[1]
    forces = formatVector[2]
    nelx, nely = formatVector[3], formatVector[4]
    Youngs, C_max, S_max = formatVector[5], formatVector[6], formatVector[7]

    x = np.linspace(0,2,nelx+1)
    y = np.linspace(0,1,nely+1)
    X,Y = np.meshgrid(x,y)

    def dist(num):
        return np.sqrt((X-circles[0][num])**2 + (Y-circles[1][num])**2) - radii[num]

    circleImage = np.minimum(dist(0),np.minimum(dist(1),dist(2)))
    circleImage = np.where(circleImage >= 0, 0,1)

    circleImage = np.reshape(circleImage.T,(nelx+1,nely+1,1))

    res = min(nelx,nely)

    forceImageX = np.zeros((nelx+1,nely+1,1))
    forceImageY = np.zeros((nelx+1,nely+1,1))
    for i in range(3):
        fx = forces[0][i] / FORCE_NORMILIZATION_FACTOR
        fy = forces[1][i] / FORCE_NORMILIZATION_FACTOR
        x_coord = int(circles[0][i] * res)
        y_coord = int(circles[1][i] * res)
        forceImageX[x_coord,y_coord,0] = fx
        forceImageY[x_coord,y_coord,0] = fy

        
    #print("Y.shape:",Y.shape)

    Y_image = (Youngs / YOUNGS_MODULUS_NORMILIZATION_FACTOR )*np.ones((nelx+1,nely+1,1))
    c_max_image = (C_max / COMPLIANCE_MAX_NORMILIZATION_FACTOR )*np.ones((nelx+1,nely+1,1))
    s_max_image = (S_max / STRESS_MAX_NORMILIZATION_FACTOR )*np.ones((nelx+1,nely+1,1))

    print("circleImage.shape:",circleImage.shape)
    print("forceImageX.shape:",forceImageX.shape)
    print("forceImageY.shape:",forceImageY.shape)
    print("Y_image.shape:",Y_image.shape)
    print("c_max_image.shape:",c_max_image.shape)
    print("s_max_image.shape:",s_max_image.shape)

    loadCondtionsImage = np.concatenate([circleImage,forceImageX,forceImageY,Y_image,c_max_image,s_max_image],axis=2)
    loadCondtionsImage = np.reshape(loadCondtionsImage,(1,nelx+1,nely+1,6))
    startBlock = np.ones((1,nelx+1,nely+1,1))
    return loadCondtionsImage,startBlock

def iteratePart(model,formatVector,numIterations:int=50):
    formattedImage,StartingBlock = formatDataForModel(formatVector)

    numImages = numIterations

    ImageToPredict = StartingBlock
    PredictedImages = [StartingBlock]

    start = time()
    for i in range(numImages):
        #use the output of the last iteration as the input for the next iteraion
        output = model.predict({'x':ImageToPredict,'loadConditions':formattedImage},verbose = 0)
        ImageToPredict = output#[0]
        PredictedImages.append(ImageToPredict)
    end = time()

    print("{} iterations took {:.2f} seconds or about {:.5f} seconds per iteration.".format(numImages,end-start,(end-start)/numImages))
    return PredictedImages

def scoreModel(TrueDataFile,model):
    trueFormatVector,TruePart,converged = loadFenicPart(TrueDataFile)
    nelx,nely = trueFormatVector[3],trueFormatVector[4]
    start = time()
    predictions = iteratePart(model,trueFormatVector)

    
    # predictionScores = []
    # for part in predictions:
    #     sol = np.reshape(part,(nelx+1,nely+1),order='F')
    #     predictionScores.append()
    x_sol = np.reshape(predictions[-1],(nelx+1,nely+1),order='F')
    converged = False
    #solution_list, objective_list, derivative_list, C_max, S_max, converged = convergenceTester(trueFormatVector, x_sol)
    end = time()
    print("\n\nDone:{}:{}".format(converged,end-start))
    fig,ax = plt.subplots(2)
    ax[0].imshow(np.reshape(TruePart,(nelx+1,nely+1),order='F').T)
    ax[1].imshow(np.reshape(predictions[-1],(nelx+1,nely+1),order='F').T)
    





def main():
    nelx = 100
    nely = nelx//2#50
    model = getModel(nelx,nely)
    path = r"\\wsl.localhost\Ubuntu\home\group2\Alienware Agents Set 7\100_50"
    dirList = os.listdir(path)

    i = 123
    DataFile = os.path.join(path,dirList[i])

    scoreModel(DataFile,model)


    #convergenceTester(problemConditions, x_sol)




