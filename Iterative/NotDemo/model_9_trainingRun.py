import tensorflow as tf
#from sklearn.model_selection import train_test_split

import numpy as np
import os
import json

from GetMassData import *
from ModelData import *

FORCE_NORMILIZATION_FACTOR = 7000
YOUNGS_MODULUS_NORMILIZATION_FACTOR = 238000000000
COMPLIANCE_MAX_NORMILIZATION_FACTOR = 0.03
STRESS_MAX_NORMILIZATION_FACTOR = 15000000

class TopOptSequence:
    def __init__(self,ID,formatted,x_array,numIterations,converged):
        self.ID = ID
        self.loadCondtions = formatted
        self.xPhys_array = x_array
        self.numIterations = numIterations
        self.nelx = self.loadCondtions[3]
        self.nely = self.loadCondtions[4]
        self.converged = converged
    
    def formatLoadCondtions(self):
        circles = self.loadCondtions[0]
        radii = self.loadCondtions[1]
        forces = self.loadCondtions[2]
        nelx, nely = self.loadCondtions[3], self.loadCondtions[4]
        Youngs, C_max, S_max = self.loadCondtions[5], self.loadCondtions[6], self.loadCondtions[7]

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

        # print("circleImage.shape:",circleImage.shape)
        # print("forceImageX.shape:",forceImageX.shape)
        # print("forceImageY.shape:",forceImageY.shape)
        # print("Y_image.shape:",Y_image.shape)
        # print("c_max_image.shape:",c_max_image.shape)
        # print("s_max_image.shape:",s_max_image.shape)

        loadCondtionsImage = np.concatenate([circleImage,forceImageX,forceImageY,Y_image,c_max_image,s_max_image],axis=2)
        return loadCondtionsImage

    def dispenceFirstIteration(self,iterationdepth:int=5,step:int=5):
        StartingBlock = np.reshape(self.xPhys_array[0],(self.nelx+1,self.nely+1,1),order='F')
        outputParts = []
        for i in range(iterationdepth):
            
            jumpIndex = min(self.numIterations-1,step*(i+1))
            outputParts.append(np.reshape(self.xPhys_array[jumpIndex],(self.nelx+1,self.nely+1,1),order='F'))

        
        formattedImage = self.formatLoadCondtions()

        return StartingBlock,formattedImage,outputParts

    def dispenceIteration(self,iterationNumber,iterationdepth:int=5,step:int=5):
        iterationNumber = min(iterationNumber,self.numIterations-1)
        StartingBlock = np.reshape(self.xPhys_array[iterationNumber],(self.nelx+1,self.nely+1,1),order='F')
        outputParts = []
        for i in range(iterationdepth):
            
            jumpIndex = min(self.numIterations-1,iterationNumber + step*(i+1))
            outputParts.append(np.reshape(self.xPhys_array[jumpIndex],(self.nelx+1,self.nely+1,1),order='F'))

        
        formattedImage = self.formatLoadCondtions()

        return StartingBlock,formattedImage,outputParts


def buildDataSet(indexesToGrab:list,path,dir_list):

    # Constants of interest
    # DATA_FILE_PATH = path to agent files
    # dir_List = all agent files
    # max_data_points = total number of datapoints

    

    #randomize the data grabed so that the first thee datapoints aren't always in the data.
    
    nonConvergedCounter = 0
    numPoints = len(indexesToGrab)

    sequenceData = []
    print("Retreiving {} Datapoints.".format(numPoints))

    for i in indexesToGrab:
        #print("{:.2f}%\t\t".format((100*(i/numPoints))),end='\r')
        try:
            #join the data file path to a random sorted member within the data directory
            pathToAgent = os.path.join(path,dir_list[i])
            formated,x_array,derivatives_array,objectives_array,markName = getData(pathToAgent)
            
        except:
            #if an exception occurs list it and move forward
            print("Exception Occured at file '{}'.".format(os.path.join(path,dir_list[i])))
            continue
        else:
            cvrg = True
            if('NotConverged' in markName):
                print("file {} has not converged.".format(dir_list[i]))
                nonConvergedCounter += 1
                cvrg = False
            else:
                #if no error occured append that data to the data list
                sequenceData.append(TopOptSequence(i,formated,x_array,len(x_array),cvrg))

    #print("100%\t\t")
    print(f"Out of {numPoints} data points gathered, {100*(nonConvergedCounter/numPoints)}% had not converged for a total of {nonConvergedCounter}")
    return sequenceData
        

def getModel():
    modelNum = 9
    model = Model_m9()
    fileSaveName = "Model_m{}".format(modelNum)
    
    
    print("Getting model {}".format(fileSaveName))
    modelPath = os.path.join(os.getcwd(), 'MachineLerning','ModelSave',fileSaveName)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(modelPath,fileSaveName),
                                                     save_weights_only=True,
                                                     verbose=1)
    
    print("Compiling Model")
    model.compile(  optimizer='Adam',
                    loss= tf.keras.losses.BinaryCrossentropy())

    if(os.path.isdir(modelPath)):
        try:
            print("Getting model weights")
            model.load_weights(os.path.join(modelPath,fileSaveName))
        except:
            print("Model weights could not be loaded.")
        else:
            print("Model weights Loaded")
    else:
        os.mkdir(modelPath)
        print("Model path created")

    
    
    return model,cp_callback

def trainModel(model,callback,data,iterationJump:int=5,pretrain:bool=False):
    def createDataset():
        loadCondtions = []
        parts = []
        outputs = []
        for i in range(len(data)):
            if(data[i].converged):
                for j in range(data[i].numIterations):
                    StartingBlock,formattedImage,outputParts = data[i].dispenceIteration(j,5,iterationJump)
                    loadCondtions.append(formattedImage)
                    parts.append(StartingBlock)
                    outputArrays = []
                    for outputBlock in outputParts:
                        outputArrays.append(outputBlock)
                    outputs.append(outputArrays)
                    if(pretrain == True and j > 0):
                        break
        
        loadCondtions = np.array(loadCondtions)
        parts = np.array(parts)
        outputs = np.array(outputs)
        return loadCondtions,parts,outputs
    
    
    
    format_array,x_array,outputs_array = createDataset()

    x1 = outputs_array[:,0,:,:,:]
    x2 = outputs_array[:,1,:,:,:]
    x3 = outputs_array[:,2,:,:,:]
    x4 = outputs_array[:,3,:,:,:]
    x5 = outputs_array[:,4,:,:,:]

    #print("format_array.shape:",format_array.shape)
    #print("x_array.shape:",x_array.shape)
    #print("outputs_array.shape:",outputs_array.shape)
    #print("x1.shape:",x1.shape)
    #print("x5.shape:",x5.shape)
    numEpochs = 10
    BatchSize = 32 # default tensorflow batchsize
    numBatches = len(x_array) // BatchSize
    BatchesPerEpoch = numBatches// numEpochs
    print("Pretraining model over {} epochs.\n\tnumSamples: {}\n\tnumBatches: {}\n\tBatches per Epoch:{}\n".format(numEpochs,len(x_array),numBatches,BatchesPerEpoch))
    
    history1 = model.fit(
        x={'x':x_array,'loadConditions':format_array},
        y=(x1,x2,x3,x4,x5),
        validation_split = 0.1,
        epochs=numEpochs,
        shuffle=True,
        steps_per_epoch = BatchesPerEpoch,
        callbacks = [callback])

    return history1

def saveHistory(train,i):
    
    # pretrain_history_dict = pretrain.history
    # name = "PretrainHistory_{}".format(i)
    # json.dump(pretrain_history_dict,open(name,'w'))
        
    
   
    train_history_dict = train.history
    name = "trainHistory_{}".format(i)
    json.dump(train_history_dict,open(name,'w'))




def main():
    dataDirectory = os.path.join("E:\TopoptGAfileSaves","Mass minimization","JustMirrored","Agents")
    DATA_FILE_PATH = os.path.join(dataDirectory,'100_50')

    dir_list = os.listdir(DATA_FILE_PATH)
    max_data_points = len(dir_list)
    print("Number of data points: {}".format(len(dir_list)))
    indexesList = np.arange(max_data_points)
    np.random.shuffle(indexesList)
    MAX_BATCH_SIZE = 50

    model,callback = getModel()
    pretrainHistory = []
    trainHistory = []

    print("Starting Batched Training")
    for BatchNumber in range(max_data_points//MAX_BATCH_SIZE):
        startIndex = BatchNumber*MAX_BATCH_SIZE
        endIndex = (BatchNumber+1)*MAX_BATCH_SIZE
        indexesForCurrentBatch = indexesList[startIndex:endIndex]
        dataSet = []
        dataSet = buildDataSet(indexesForCurrentBatch,DATA_FILE_PATH,dir_list)
        #print(len(indexesForCurrentBatch),startIndex,endIndex,len(dataSet))

        #pretrainHistory = trainModel(model,callback,dataSet,5,pretrain=True)
        trainHistory = trainModel(model,callback,dataSet,10,pretrain=False)

        saveHistory(trainHistory,BatchNumber)

def cleanData():
    dataDirectory = os.path.join("E:\TopoptGAfileSaves","Mass minimization","correctFOrmat")
    path = dataDirectory#os.path.join(dataDirectory,'100_50')
    AgentsToGrab = os.listdir(path)
    numAgents = len(AgentsToGrab)
    print("There are {} files to explore.".format(numAgents))

    invalidAgents = []
    for i,agent in enumerate(AgentsToGrab):
        print("{:.1f}%\t".format(100*(i/numAgents)),end='\r')
        agentFiles = os.listdir(os.path.join(path,agent))
        for fileName in agentFiles:
            if(('Invalid' in fileName) or ('NotConverged' in fileName)):
                invalidAgents.append(agent)
                break
    print("100%\t\nOf the {} files scanned {} were invalid.".format(numAgents,len(invalidAgents)))
    if(input("ENTER(y/n):") == 'Y'):   

        for agent in invalidAgents:
            agentFiles = os.listdir(os.path.join(path,agent))
            for fileName in agentFiles:
                fileToRemove = os.path.join(path,agent,fileName)
                #print("removing: {}".format(fileToRemove))
                os.remove(os.path.join(path,agent,fileName))
            directoryToRemove = os.path.join(path,agent)
            print("removing: {}".format(directoryToRemove))
            os.rmdir(os.path.join(path,agent))
    print("\nDone.")
            

def getFormatStats():
    dataDirectory = os.path.join("E:\TopoptGAfileSaves","Mass minimization","JustMirrored","Agents")
    path = dataDirectory#os.path.join(dataDirectory,'100_50')
    AgentsToGrab = os.listdir(path)
    numAgents = len(AgentsToGrab)
    print("There are {} files to explore.".format(numAgents))

    fx_array = []
    fy_array = []
    for i,agent in enumerate(AgentsToGrab):
        print("{:.1f}%\t".format(100*(i/numAgents)),end='\r')
        agentFiles = os.listdir(os.path.join(path,agent))
        for fileName in agentFiles:
            if(('Invalid' in fileName) or ('NotConverged' in fileName)):
                invalidAgents.append(agent)
                break
    print("100%\t\nOf the {} files scanned {} were invalid.".format(numAgents,len(invalidAgents)))
    if(input("ENTER(y/n):") == 'Y'):   

        for agent in invalidAgents:
            agentFiles = os.listdir(os.path.join(path,agent))
            for fileName in agentFiles:
                fileToRemove = os.path.join(path,agent,fileName)
                #print("removing: {}".format(fileToRemove))
                os.remove(os.path.join(path,agent,fileName))
            directoryToRemove = os.path.join(path,agent)
            print("removing: {}".format(directoryToRemove))
            os.rmdir(os.path.join(path,agent))
    print("\nDone.")
            


if(__name__=="__main__"):
    main()
    #cleanData()


