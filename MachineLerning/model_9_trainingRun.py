import tensorflow as tf
#from sklearn.model_selection import train_test_split

import numpy as np
import os
import json
from time import sleep
import shutil

from GetMassData import *
from ModelData import *


def buildDataSet(indexesToGrab:list,path,dir_list):
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
            # elif(len(x_array) < 20):
            #     print("Too few Iterations")
            #     nonConvergedCounter += 1
            #     cvrg = False
            else:
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
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=100000,decay_rate=0.9,staircase=False)
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

def trainModel(model,callback,data,iterationJump:int=5,pretrain:bool=False,validationData:dict=None):
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

                # 50% chance to drop some samples past iteration 1, 
                if(j > 1 and np.random.random() < .50):
                    loadCondtions.append(formattedImage)
                    parts.append(StartingBlock)
                    outputArrays = []
                    for outputBlock in outputParts:
                        outputArrays.append(outputBlock)
                    outputs.append(outputArrays)
                
                #break conditions in case the data is not relevant
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
    numEpochs = 5
    BatchSize = 32 # default tensorflow batchsize
    numBatches = max(1,len(x_array) // BatchSize)
    BatchesPerEpoch = max(1,numBatches// numEpochs)
    print("Training model over {} epochs.\n\tnumSamples: {}\n\tnumBatches: {}\n\tBatches per Epoch:{}\n".format(numEpochs,len(x_array),numBatches,BatchesPerEpoch))

    if(validationData == None):
        history1 = model.fit(
            x={'x':x_array,'loadConditions':format_array},
            y=(x1,x2,x3,x4,x5),
            validation_split = 0.1,
            epochs=numEpochs,
            shuffle=True,
            steps_per_epoch = BatchesPerEpoch, 
            callbacks = [callback])
    else:
        history1 = model.fit(
            x={'x':x_array,'loadConditions':format_array},
            y=(x1,x2,x3,x4,x5),
            validation_data = ({'x':validationData['x'], 'loadConditions':validationData['loadConditions']} , 
                               (validationData['out1'],validationData['out2'],validationData['out3'],validationData['out4'],validationData['out5'])),
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
 
def getValidationData(path,numPoints,iterationJump:int = 5):
    """
    Return a set of arrays similar to the create dataset of the training function.
    returns data in a python dict that can be passed to the trainmodel function to use as propper validation data to score the model on.

    Will random take some number of sequences in the folder and then collect a random set of iterations from those sequences until it reaches numPoints
    """
    #seed the random so that the return values are the same each time
    rng = np.random.Generator(np.random.PCG64(0))
    dir_list = os.listdir(path)
    max_data_points = len(dir_list)
    

    indexesList = np.arange(max_data_points)
    seqences = buildDataSet(indexesList,path,dir_list)

    iterationsPerSequence = numPoints // len(seqences)

    loadCondtions = []
    parts = []
    outputs = []
    for i in range(len(seqences)):
        chanceToSelectIteration = iterationsPerSequence / seqences[i].numIterations
        
        for j in range(seqences[i].numIterations):
            if(seqences[i].numIterations > iterationJump*5):
                StartingBlock,formattedImage,outputParts = seqences[i].dispenceIteration(j,5,iterationJump,False)
            else:
                StartingBlock,formattedImage,outputParts = seqences[i].dispenceFullPartIteraion(5,False)

            #randomly select a few points of data from the sequence to serve as data
            if(rng.random() < chanceToSelectIteration):
                loadCondtions.append(formattedImage)
                parts.append(StartingBlock)
                outputArrays = []
                for outputBlock in outputParts:
                    outputArrays.append(outputBlock)
                outputs.append(outputArrays)
            
            #break conditions in case the data is not relevant
            if(seqences[i].numIterations <= iterationJump*5):
                #if there is less iterations than the model can train over
                break
            elif(seqences[i].converged == False and j > 0):
                # if the end result of the part is invalid
                break
    
    format_array = np.array(loadCondtions)
    x_array = np.array(parts)
    outputs_array = np.array(outputs)
    
    x1 = outputs_array[:,0,:,:,:]
    x2 = outputs_array[:,1,:,:,:]
    x3 = outputs_array[:,2,:,:,:]
    x4 = outputs_array[:,3,:,:,:]
    x5 = outputs_array[:,4,:,:,:]

    print("The validation data has {} samples.".format(len(x_array)))

    dictOfValidationData = {'loadConditions':format_array, 'x':x_array, 'out1':x1, 'out2':x2, 'out3':x3, 'out4':x4, 'out5':x5} 
    return dictOfValidationData
 
def saveHistory(train,i):
    
    # pretrain_history_dict = pretrain.history
    # name = "PretrainHistory_{}".format(i)
    # json.dump(pretrain_history_dict,open(name,'w'))
        
    
   
    train_history_dict = train.history
    name = "trainHistory_{}".format(i)
    json.dump(train_history_dict,open(name,'w'))

def moveModelCheckPoint():
    """
    Takes the model folder found in the current saving directory and copies the entire folder to an external folder.
    """
    modelPath = os.path.join(os.getcwd(), 'MachineLerning','ModelSave',"Model_m9")
    externalFolder = r'E:\TopoptGAfileSaves\Models\Model_m9\AutoSaves'

    saveName = 'Model_m9_save_'
    currentSaves = os.listdir(externalFolder)
    currentNum = len(currentSaves) + 1

    #create new directory
    pathToSaveTo = os.path.join(externalFolder,saveName + str(currentNum))
    try:
        os.makedirs(pathToSaveTo,exist_ok=False)
    except:
        print("Error in creating directory, folder already exists!.\n")
        return
    else:
        for modelCheckPoint in os.listdir(modelPath):
            fullFileName = os.path.join(modelPath,modelCheckPoint)
            if(os.path.isfile(fullFileName)):
                shutil.copy2(fullFileName,pathToSaveTo)




def main():
    nelx = 100
    nely = nelx//2#50

    #dataDirectory = os.path.join("E:\TopoptGAfileSaves","Mass minimization")
    dataDirectory = r"E:\TopoptGAfileSaves\Mass minimization\AlienWareData\Augmented\set5\Agents"
    DATA_FILE_PATH = os.path.join(dataDirectory,'{}_{}'.format(nelx,nely))

    dir_list = os.listdir(DATA_FILE_PATH)
    max_data_points = len(dir_list)
    print("Number of data points: {}".format(len(dir_list)))

    indexesList = np.arange(max_data_points)
    np.random.shuffle(indexesList)


    MAX_BATCH_SIZE = 100
    MAX_BATCH_SIZE = min(MAX_BATCH_SIZE,max_data_points)

    model,callback = getModel(nelx+1,nely+1)

    validationDict = getValidationData(r'E:\TopoptGAfileSaves\Mass minimization\Training Validation\100_50',100,5)

    numBatches = max((max_data_points//MAX_BATCH_SIZE) + 1,1)
    print("Starting Batched Training with {} super batches.".format(numBatches))
    for BatchNumber in range(0,numBatches):

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
            trainHistory = trainModel(model,callback,dataSet,iterationJump=5,pretrain=False,validationData=validationDict)

            saveHistory(trainHistory,BatchNumber)
        
        if ((BatchNumber > 0) and (BatchNumber %3 == 0)):
            #save the current model to a new folder
            moveModelCheckPoint()
    
    print("Done.")

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
    #moveModelCheckPoint()


