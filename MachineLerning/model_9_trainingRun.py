import tensorflow as tf
#from sklearn.model_selection import train_test_split

import numpy as np
import os
import json
from time import sleep

from GetMassData import *
from ModelData import *


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
            elif(len(x_array) < 20):
                print("Too few Iterations")
                nonConvergedCounter += 1
                cvrg = False
            #else:
                #if no error occured append that data to the data list
            sequenceData.append(TopOptSequence(i,formated,x_array,len(x_array),cvrg))

    #print("100%\t\t")
    print(f"Out of {numPoints} data points gathered, {100*(nonConvergedCounter/numPoints)}% had not converged for a total of {nonConvergedCounter}")
    return sequenceData
        
def getModel(resX:int=101,resY:int=51):
    modelNum = 9
    model = Model_m9(resX,resY)
    fileSaveName = "Model_m{}".format(modelNum)
    
    
    print("Getting model {}".format(fileSaveName))
    modelPath = os.path.join(os.getcwd(), 'MachineLerning','ModelSave',fileSaveName)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(modelPath,fileSaveName),
                                                     save_weights_only=True,
                                                     verbose=1)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.00001,decay_steps=100000,decay_rate=0.9,staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    print("Compiling Model")
    model.compile(  optimizer=optimizer,
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
            
            for j in range(data[i].numIterations):
                if(data[i].numIterations > iterationJump*5):
                    StartingBlock,formattedImage,outputParts = data[i].dispenceIteration(j,5,iterationJump,False)
                else:
                    StartingBlock,formattedImage,outputParts = data[i].dispenceFullPartIteraion(5,False)
                loadCondtions.append(formattedImage)
                parts.append(StartingBlock)
                outputArrays = []
                for outputBlock in outputParts:
                    outputArrays.append(outputBlock)
                outputs.append(outputArrays)
                if(pretrain == True and j > 0):
                    #if we only want the first iteration set
                    break
                elif(data[i].numIterations <= iterationJump*5):
                    #if there is less iterations than the model can train over
                    break
                elif(data[i].converged == False and j > 0):
                    # if the end result of the part is invalid
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
    print("Training model over {} epochs.\n\tnumSamples: {}\n\tnumBatches: {}\n\tBatches per Epoch:{}\n".format(numEpochs,len(x_array),numBatches,BatchesPerEpoch))
    history1 = model.fit(
        x={'x':x_array,'loadConditions':format_array},
        y=(x1,x2,x3,x4,x5),
        validation_split = 0.1,
        epochs=numEpochs,
        shuffle=True,
        steps_per_epoch = BatchesPerEpoch, 
        callbacks = [callback])

    return history1

def pretrainModel(model,callback,data):
    def createDataset():
        loadCondtions = []
        parts = []
        outputs = []
        for i in range(len(data)):
            if(data[i].converged):
                
                StartingBlock,formattedImage,outputParts = data[i].dispenceFullPartIteraion(5,True)
                loadCondtions.append(formattedImage)
                parts.append(StartingBlock)
                outputArrays = []
                for outputBlock in outputParts:
                    outputArrays.append(outputBlock)
                outputs.append(outputArrays)
        
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
    numEpochs = 30
    BatchSize = 32 # default tensorflow batchsize
    numBatches = len(x_array) // BatchSize
    BatchesPerEpoch = max(1,numBatches// numEpochs)
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
    #dataDirectory = os.path.join("E:\TopoptGAfileSaves","Mass minimization")
    dataDirectory = r"E:\TopoptGAfileSaves\Mass minimization\AlienWareData\Augmented\Set1\Agents"
    DATA_FILE_PATH = os.path.join(dataDirectory,'120_60')

    dir_list = os.listdir(DATA_FILE_PATH)
    max_data_points = len(dir_list)
    print("Number of data points: {}".format(len(dir_list)))
    indexesList = np.arange(max_data_points)
    np.random.shuffle(indexesList)
    MAX_BATCH_SIZE = 50
    MAX_BATCH_SIZE = min(MAX_BATCH_SIZE,max_data_points)

    model,callback = getModel(121,61)

    print("Starting Batched Training")
    numBatches = max((max_data_points//MAX_BATCH_SIZE) + 1,1)
    for BatchNumber in range(16,numBatches):

        print("Batch: {}".format(BatchNumber))
        startIndex = BatchNumber*MAX_BATCH_SIZE
        endIndex = min((BatchNumber+1)*MAX_BATCH_SIZE,max_data_points-1)
        if(startIndex >= endIndex):
            break
        else:
            indexesForCurrentBatch = indexesList[startIndex:endIndex]
            dataSet = []
            dataSet = buildDataSet(indexesForCurrentBatch,DATA_FILE_PATH,dir_list)
            print(len(indexesForCurrentBatch),startIndex,endIndex,len(dataSet))

            #pretrainHistory = trainModel(model,callback,dataSet,5,pretrain=True)
            trainHistory = trainModel(model,callback,dataSet,iterationJump=10,pretrain=False)

            saveHistory(trainHistory,BatchNumber)
            sleep(600.0)#let my computer sleep for a few minutes before training again

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
            


if(__name__=="__main__"):
    main()
    #cleanData()


