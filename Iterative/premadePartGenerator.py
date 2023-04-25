import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tensorflow as tf
import os
from time import time
from fenics_tester import *

FORCE_NORMILIZATION_FACTOR = 7000
YOUNGS_MODULUS_NORMILIZATION_FACTOR = 238000000000
COMPLIANCE_MAX_NORMILIZATION_FACTOR = 0.03
STRESS_MAX_NORMILIZATION_FACTOR = 15000000


class ConcatAndCrop(tf.keras.layers.Layer):
    """
    Takes the images from the upscaled image and the image from the skip connection and concatedates the two by cropping the upscaled image to the size of the skip connection
    """
    def __init__(self,upscale_shape,skipConnection_shape):
        super(ConcatAndCrop, self).__init__()
        #left crop the image by the difference of the two shapes
        #the upscale shape will aways be the larger of the two

        horizontalCrop = upscale_shape[1] - skipConnection_shape[1]
        verticalCrop = upscale_shape[2] - skipConnection_shape[2]
        print(f"concat shapes {upscale_shape} and {skipConnection_shape} create padding {horizontalCrop} and {verticalCrop}")

        self.cropLayer = tf.keras.layers.Cropping2D(cropping=((horizontalCrop,0),(verticalCrop,0)))
        self.concatLayer = tf.keras.layers.Concatenate()
        #print(f"cropped to {self.cropLayer.output_shape} and {skipConnection_shape} ")

    def call(self,upscaledLayer,skipConnectionLayer):

        crop = self.cropLayer(upscaledLayer)
        concat = self.concatLayer([crop,skipConnectionLayer])
        return concat

def buildModel_m9(x_inputShape = (101,51,1),LoadConditionsImage = (101,51,6),activation='relu'):
    partInput = tf.keras.layers.Input(shape=x_inputShape,name="x")
    loadsInput = tf.keras.layers.Input(shape=LoadConditionsImage,name="loadConditions")

    concatenatedStartLayer = tf.keras.layers.Concatenate()([partInput,loadsInput])

    #First Convolution Layer
    conv_128_64 = tf.keras.layers.Conv2D(filters= 16, kernel_size=(3,3),padding='same',activation=activation)(concatenatedStartLayer)
    conv_128_64 = tf.keras.layers.Conv2D(filters= 16, kernel_size=(3,3),padding='same',activation=activation)(conv_128_64)
    conv_64_32 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv_128_64)
    conv_64_32 = tf.keras.layers.GaussianNoise(stddev=0.1)(conv_64_32)

    #Second convolution Layer
    conv_64_32 = tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3),padding='same',activation=activation)(conv_64_32)
    conv_64_32 = tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3),padding='same',activation=activation)(conv_64_32)
    conv_32_16 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv_64_32)
    conv_32_16 = tf.keras.layers.GaussianNoise(stddev=0.1)(conv_32_16)

    conv_32_16 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3),padding='same',activation=activation)(conv_32_16)
    conv_32_16 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3),padding='same',activation=activation)(conv_32_16)
    conv_16_8 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv_32_16)
    conv_16_8 = tf.keras.layers.GaussianNoise(stddev=0.1)(conv_16_8)

    conv_16_8 = tf.keras.layers.Conv2D(filters= 128, kernel_size=(3,3),padding='same',activation=activation)(conv_16_8)
    conv_16_8 = tf.keras.layers.Conv2D(filters= 128, kernel_size=(3,3),padding='same',activation=activation)(conv_16_8)

    #upscaleLayer
    #upscaling is performed by convolution transpose where stride=2 < kernalsize
    convUpscale_32_16 = tf.keras.layers.Conv2DTranspose(filters= 64, kernel_size=(5,5),strides=2,padding='same',activation=activation)(conv_16_8)
    convUpscale_32_16 = tf.keras.layers.GaussianNoise(stddev=0.1)(convUpscale_32_16)
    convUpscale_32_16 = ConcatAndCrop(convUpscale_32_16.shape,conv_32_16.shape)(convUpscale_32_16,conv_32_16)
    convUpscale_32_16 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_32_16)
    convUpscale_32_16 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_32_16)

    convUpscale_64_32 = tf.keras.layers.Conv2DTranspose(filters= 32, kernel_size=(5,5),strides=2,padding='same',activation=activation)(convUpscale_32_16)
    convUpscale_64_32 = tf.keras.layers.GaussianNoise(stddev=0.1)(convUpscale_64_32)
    convUpscale_64_32 = ConcatAndCrop(convUpscale_64_32.shape,conv_64_32.shape)(convUpscale_64_32,conv_64_32)
    convUpscale_64_32 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_64_32)
    convUpscale_64_32 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_64_32)

    convUpscale_128_64 = tf.keras.layers.Conv2DTranspose(filters= 64, kernel_size=(5,5),strides=2,padding='same',activation=activation)(convUpscale_64_32)
    convUpscale_64_32 = tf.keras.layers.GaussianNoise(stddev=0.1)(convUpscale_64_32)
    convUpscale_128_64 = ConcatAndCrop(convUpscale_128_64.shape,conv_128_64.shape)(convUpscale_128_64,conv_128_64)
    convUpscale_128_64 = tf.keras.layers.Conv2D(filters = 16, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_128_64)
    convUpscale_128_64 = tf.keras.layers.Conv2D(filters = 16, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_128_64)

    output_part = tf.keras.layers.Conv2D(filters= 1, kernel_size=(1,1),padding='same',activation='hard_sigmoid', name="x_out")(convUpscale_128_64)
    """
    The hard sigmoid activation, defined as:
        if x < -2.5: return 0
        if x > 2.5: return 1
        if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
    """

    return tf.keras.Model(inputs= [partInput,loadsInput],outputs=[output_part])#,finishedOutput])

class Model_m9(tf.keras.Model):
    def __init__(self,nelx:int=101,nely:int=51):

        super(Model_m9, self).__init__()

        self.model = buildModel_m9((nelx,nely,1),(nelx,nely,6))

        

    def call(self,data,training = False):
        #part = data['x']
        #forces = data['forces']
        #supports = data['supports']
        #filled = data['filled']
        #print(1)
        if(training):
            
            out1 = self.model(data)
            data['x'] = out1
            
            #print(2)
            out2 = self.model(data)
            data['x'] = out2

            out3 = self.model(data)
            data['x'] = out3
        
            out4 = self.model(data)
            data['x'] = out4

            out5 = self.model(data)
            #print(6)
            return out1,out2,out3,out4,out5
        else:
            return self.model(data)

def fenics_testPart(formatted,part):
    return 0,0

def scoreModelPredictions(formatted,part_list):
    """
    Take the formatted array and a part and return the current mass, stress, and compliance of the part.
    
    Uses fenics to generate the stress and compliance scores
    """

    c_max = formatted[6]
    s_max = formatted[7]
    scoreList = []

    for part in part_list:
        print("\nTesting part")
        part_flat = np.ravel(part,order='F')

        #print("\tCompliance Max: {}".format(c_max))
        #print("\tStress Max: {}".format(s_max))

        compliance,stress = convergenceTester(formatted,part_flat,1)
        mass = np.sum(part_flat)

        print("\tCompliance: {}".format(compliance))
        print("\tStress: {}".format(stress))
        # print("\tMass: {}".format(mass))
        #print("Part has been tested\n")
        #stress = np.max(stress)

        stress -= s_max
        compliance -= c_max
        # print("Scaling Values")
        # print("\tCompliance: {}".format(compliance))
        # print("\tStress: {}".format(stress))

        stress /= STRESS_MAX_NORMILIZATION_FACTOR
        compliance /= COMPLIANCE_MAX_NORMILIZATION_FACTOR
        #print("Scaling Values")
        #print("\tCompliance: {}".format(compliance))
        #print("\tStress: {}".format(stress))

        

        score = mass + np.exp(compliance) + np.exp(stress)
        scoreList.append(score)

    return scoreList

def doublePartSize(part,formatVector):
    nelx, nely = formatVector[3], formatVector[4]
    nelx *= 2
    nely *= 2

    newPart = np.zeros((nelx+1,nely+1))
    for x in range(nelx//2):
        for y in range(nely//2):
            indexes = [[2*x,2*y],[2*x + 1,2*y],[2*x,2*y + 1],[2*x + 1,2*y + 1]]
            val = part[x,y]
            for x1,y1 in indexes:

                newPart[x1,y1] = val
    
    return newPart

def calcRatio(a, b):
        """
        Finds the ratio between two numbers. Used to prevent FEniCS from freaking out.
        Aka, in xDim, yDim, and L, W within massopt_n.py
        """
        gcd = np.gcd(a, b)

        aReduced = a / gcd
        bReduced = b / gcd
        
        return aReduced, bReduced

def plotFormatVector(formatVector,res:int=100):
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
        
    plt.show()

def SaveAsGif(images,nelx,nely,name:str="out"):
    try:
        from PIL import Image
        import io
    except ImportError:
        print("Propper modules not installed.")
        return
    else:
        print("Saving iterations as Gif.")
        imageArray = []
        for i,image in enumerate(images):
            print("{:.2f}%\t".format(100*(i/len(images))),end='\r')
            fig,ax = plt.subplots(1,1)
            
            if(i == 0):
                ax.set_title("Iteration: {}".format(i))
            else:
                im1 = np.reshape(image,((nelx+1)*(nely+1)))
                im2 = np.reshape(images[i-1],((nelx+1)*(nely+1)))
                ax.set_title("Iteration: {}, Change: {:.5f}".format(i,np.linalg.norm(im1-im2,ord=np.inf)))
            ax.imshow(np.reshape(image,(nelx+1,nely+1)).T,cmap='seismic',norm=colors.Normalize(vmin=0,vmax=1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')

            im = Image.open(img_buf)
            imageArray.append(im)

            #plt.show()
        print("100%         \nDone.")
        im = imageArray[0]
        imageArray.pop(0)
        im.save(str(name) + ".gif".format(i),save_all=True,append_images = imageArray,optimize=False,loop=1)
        im.close()
    
def loadFenicsPart(agentFileToGet):
    """
    Grabs and unpacks the data stored inside an agent file.
    To be used in conjunction with the fenics data creation and the saving format above.

    Returns:
        - format vector
        - part
        - converged
    """

    #print(agentFileToGet)
    FilesToGrab = os.listdir(agentFileToGet)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []
    markName = ''


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(agentFileToGet,fileName))
            #print('loadCondtions Exist')
            markIndex = fileName.index('loadConditions')
            if(markIndex > 0):
                markName = fileName[:markIndex-1]
            
        elif('iteration_' in fileName):
            numberStart = fileName.index('iteration_')
            number_extension = fileName[numberStart+len('iteration_'):]
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
        x,obj,der = unpackIterations(arrays)
        x_array.append(x)
        derivatives_array.append(der)
        objectives_array.append(obj)
    
    #print(markName)
    converged = True
    if('NotConverged' in markName):
        converged = False
    
    return formated,x_array[-1],converged

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

    # print("circleImage.shape:",circleImage.shape)
    # print("forceImageX.shape:",forceImageX.shape)
    # print("forceImageY.shape:",forceImageY.shape)
    # print("Y_image.shape:",Y_image.shape)
    # print("c_max_image.shape:",c_max_image.shape)
    # print("s_max_image.shape:",s_max_image.shape)

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

def getModelPrediction(TrueDataFile,model):
    trueFormatVector,TruePart,converged = loadFenicsPart(TrueDataFile)
    nelx,nely = trueFormatVector[3],trueFormatVector[4]

    predictions = iteratePart(model,trueFormatVector,50)

    return trueFormatVector,TruePart,predictions[-1],converged

def visualizeValidation(pathToData,pointsToGrab:int = 5):
    model = getModel(100,50)

    dataPoints = os.listdir(pathToData)
    maxDataPoints = len(dataPoints)
    pointsToGrab = min(maxDataPoints,pointsToGrab)

    indexes = np.arange(maxDataPoints,dtype='int32')
    np.random.shuffle(indexes)
    indexes = indexes[:pointsToGrab]

    preditions = []

    for i in indexes:
        dataPoint = os.path.join(pathToData,dataPoints[i])
        formatedVector,truePart,predition,converged = getModelPrediction(dataPoint,model)
        preditions.append([dataPoints[i],formatedVector,converged,np.reshape(truePart,(101,51),order='F'),np.reshape(predition,(101,51))])

    lossFunction = tf.keras.losses.binary_crossentropy
    
    fig,ax = plt.subplots(pointsToGrab,3)

    def plotFormatVector(index,formated):
        circles = formated[0]
        radii = formated[1]
        forces = formated[2]
        nelx, nely = formated[3], formated[4]
        res = nelx
        
        xDim,yDim = calcRatio(nelx,nely)
        x = np.linspace(0,xDim,res,True)
        y = np.linspace(0,yDim,res//2,True)

        X,Y = np.meshgrid(x,y)

        def dist(circleIndex):
            return np.sqrt((X-circles[0][circleIndex])**2 + (Y-circles[1][circleIndex])**2) - radii[circleIndex]
        
        circlesMap = np.minimum(dist(0),np.minimum(dist(1),dist(2)))
        circlesMap = np.where(circlesMap<=0,1,0)

        
        ax[index,0].imshow(circlesMap,cmap='gray_r')
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
            ax[index,0].plot(x1,y1)

        plotForce(0)
        plotForce(1)
        plotForce(2)

    for i in range(pointsToGrab):
        fileName,formatVector,converged,true,pred = preditions[i]
        plotFormatVector(i,formatVector)

        ax[i,0].set_title(fileName)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)

        ax[i,1].imshow(true.T,cmap='gray_r',norm=colors.Normalize(0,1))
        ax[i,1].set_title("converged={}".format(converged))
        ax[i,1].get_xaxis().set_visible(False)
        ax[i,1].get_yaxis().set_visible(False)

        ax[i,2].imshow(pred.T,cmap='gray_r',norm=colors.Normalize(0,1))

        diff = np.mean(lossFunction(true,pred))
        ax[i,2].set_title("diff={:.3f}".format(diff))
        ax[i,2].get_xaxis().set_visible(False)
        ax[i,2].get_yaxis().set_visible(False)

        print("File {}(converged={}) has difference {}".format(fileName,converged,diff))
    
    plt.show()

def shiftLoadConditions(formatted,shiftAmountLR,shiftAmountUD):

    circles = formatted[0].copy()
    radii = formatted[1].copy()
    forces = formatted[2].copy()
    nelx, nely = formatted[3], formatted[4]
    Y, C_max, S_max = formatted[5], formatted[6], formatted[7]

    x_shiftAmount = 2/nelx
    y_shiftAmount = 1/nely

    for i in range(3):
       circles[0][i] += shiftAmountLR*x_shiftAmount 
       circles[1][i] += shiftAmountUD*y_shiftAmount
    
    shiftedFormat = [circles,radii,forces,nelx,nely,Y,C_max,S_max]
    #showShiftedPart(iterationsArray,nelx+1,nely+1,shiftAmountLR,shiftAmountUD)
    return shiftedFormat


def getPadding(lastIteration,nelx,nely):
    cutOff = 0.05
    lastIteration = np.where(lastIteration >= cutOff,1,0)

    im = np.reshape(lastIteration,(nelx,nely),order='F')

    #get padding on left
    lPad = 0
    for x in range(0,nelx,1):
        currentCol = im[x,:]
        avgVal = np.mean(currentCol)
        if(avgVal == 0):
            lPad += 1
        else:
            break
        
    print(lPad)
    rPad = 0
    for x in range(nelx-1,-1,-1):
        currentCol = im[x,:]
        avgVal = np.mean(currentCol)
        if(avgVal == 0):
            rPad += 1
        else:
            break
    print(rPad)

    uPad = 0
    for y in range(0,nely,1):
        currentRow = im[:,y]
        avgVal = np.mean(currentRow)
        if(avgVal == 0):
            uPad += 1
        else:
            break
    print(uPad)

    dPad = 0
    for y in range(nely-1,-1,-1):
        currentRow = im[:,y]
        avgVal = np.mean(currentRow)
        if(avgVal == 0):
            dPad += 1
        else:
            break
    print(dPad)

    return lPad,rPad,uPad,dPad

def averageArray(ar1):
    n = ar1.shape[0]
    newArray = np.zeros(n)
    #print(n)

    for i in range(n):
        if (i == 0):
            a1 = ar1[i]
            a2 = ar1[i+1]
            a3 = 0
        elif(i==n-1):
            a1 = ar1[i]
            a2 = ar1[i-1]
            a3 = 0
        else:
            a1 = ar1[i-1]
            a2 = ar1[i]
            a3 = ar1[i+1]
        
        v = (a1+a2+a3)/3
        newArray[i] = v

    return newArray

def shiftImage(image,shiftLR,shiftUD):

    if(shiftLR < 0):
        while(shiftLR < 0):
            shiftLR += 1
            colToRepeat = image[:,0]
            #shift over image
            image = np.roll(image,-1,0)
            #image[:,0] = averageArray(colToRepeat)
    elif(shiftLR > 0):
        while(shiftLR > 0):
            shiftLR -= 1
            colToRepeat = image[:,-1]
            #shift over image
            image = np.roll(image,1,0)
            #image[:,-1] = averageArray(colToRepeat)
    
    if(shiftUD < 0):
        while(shiftUD < 0):
            shiftUD += 1
            rowToRepeat = image[0,:]
            #shift over image
            image = np.roll(image,-1,1)
            #image[0,:] = averageArray(rowToRepeat)
    elif(shiftUD > 0):
        while(shiftUD > 0):
            shiftUD -= 1
            rowToRepeat = image[-1,:]
            #shift over image
            image = np.roll(image,1,1)
            #image[-1,:] = averageArray(rowToRepeat)


    return image


def scoreValidations(pathToData,pointsToGrab:int = 100, numIterations:int = 50):
    model = getModel(100,50)

    dataPoints = os.listdir(pathToData)
    maxDataPoints = len(dataPoints)
    pointsToGrab = min(maxDataPoints,pointsToGrab)

    indexes = np.arange(maxDataPoints,dtype='int32')
    np.random.shuffle(indexes)
    indexes = indexes[:pointsToGrab]

    preditions = []
    dataTrue = []
    formatVecotrs_array = []

    partNames = []
    
    print("Collecting {} points of data.".format(pointsToGrab))
    for i in indexes:
        dataPoint = os.path.join(pathToData,dataPoints[i])
        formated,part,converged = loadFenicsPart(dataPoint)
        partNames.append([dataPoints[i],converged])
        dataTrue.append(np.reshape(part,(1,101,51),order="F"))
        formatImage,_ = formatDataForModel(formated)
        formatVecotrs_array.append(formatImage)

    truePart_array = np.concatenate(dataTrue,axis=0)
    formatVectors_array = np.concatenate(formatVecotrs_array,axis=0)
    startingBlock_array = np.ones((pointsToGrab,101,51,1))

    print(formatVectors_array.shape)
    print(startingBlock_array.shape)
    ImageToPredict = startingBlock_array

    start = time()
    for i in range(numIterations):
        #use the output of the last iteration as the input for the next iteraion
        output = model.predict({'x':ImageToPredict,'loadConditions':formatVectors_array},verbose = 0)
        ImageToPredict = output#[0]
    end = time()

    print("{} iterations took {:.2f} seconds or about {:.5f} seconds per iteration.".format(numIterations,end-start,(end-start)/numIterations))
    finalPredictedImages = np.reshape(output,(pointsToGrab,101,51))

    lossValues = tf.keras.losses.binary_crossentropy(truePart_array,finalPredictedImages,axis=(1,2))
    print(lossValues.shape)
    #lossValues = np.reshape
    #lossPerImage = np.mean(lossValues,axis=(1,2))
    #print(lossPerImage.shape)
    #print(lossPerImage)
    convergedLoss = []
    otherLoss = []

    for i in range(pointsToGrab):
        print("Part {}({}) has a loss of {:.4f}.".format(partNames[i][0],partNames[i][1],lossValues[i]))
        if(partNames[i][1]):
            convergedLoss.append(lossValues[i])
        else:
            otherLoss.append(lossValues[i])
    
    print("\nAverage loss is: {}.".format(np.mean(lossValues)))
    print("loss for converged parts is: {}.".format(np.mean(convergedLoss)))
    print("loss for unconverged parts is: {}.".format(np.mean(otherLoss)))

def iteratePartWithShift(model,formatedVector,numIterations:int=50,shiftAmnt:int=1):

    nelx = formatedVector[3]
    nely = formatedVector[4]
    shiftAmounts = []

    formattedImage_array = []
    for i in range(-shiftAmnt,shiftAmnt+1,1):
        for j in range(-shiftAmnt,shiftAmnt+1,1):
            shiftAmounts.append([i,j])
            shiftedVector = shiftLoadConditions(formatedVector.copy(),i,j)
            formattedImage,StartingBlock = formatDataForModel(shiftedVector)
            formattedImage_array.append(formattedImage)



    ImageToPredict = np.ones((len(formattedImage_array),nelx+1,nely+1,1))
    formattedImage_array = np.concatenate(formattedImage_array,axis=0)
    PredictedImages = [StartingBlock]

    start = time()
    for i in range(numIterations):
        #use the output of the last iteration as the input for the next iteraion
        output = model.predict({'x':ImageToPredict,'loadConditions':formattedImage_array},verbose = 0)
        ImageToPredict = output#[0]
        PredictedImages.append(ImageToPredict)
    end = time()

    print("{} iterations took {:.2f} seconds or about {:.5f} seconds per iteration.".format(numIterations,end-start,(end-start)/numIterations))
    return PredictedImages,shiftAmounts

def scoreOutputs(truePart,predictedPart_list):

    lossFunction = tf.keras.losses.binary_crossentropy

    lossValues = []
    for pred in predictedPart_list:
        lossValues.append(lossFunction(truePart,pred,axis=(0,1)))
    
    return lossValues

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


def fenicsAgentToGif(dataPoint):
    """
    Grabs and unpacks the data stored inside an agent file.
    To be used in conjunction with the fenics data creation and the saving format above.

    Returns:
        - format vector
        - part
        - converged
    """

    #print(agentFileToGet)
    FilesToGrab = os.listdir(dataPoint)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []
    markName = ''


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(dataPoint,fileName))
            #print('loadCondtions Exist')
            markIndex = fileName.index('loadConditions')
            if(markIndex > 0):
                markName = fileName[:markIndex-1]
            
        elif('iteration_' in fileName):
            numberStart = fileName.index('iteration_')
            number_extension = fileName[numberStart+len('iteration_'):]
            extesionIndex = number_extension.find('.')
            number = int(number_extension[:extesionIndex])
            #print(number)
            iterations.append([number,np.load(os.path.join(dataPoint,fileName))])
        #print(fileName)
    
    def sortKey(x):
        return x[0]

    iterations.sort(key=sortKey)

    formated = unpackLoadConditions(loadConditions)
    nelx,nely = formated[3],formated[4]
    x_array = []
    derivatives_array = []
    objectives_array = []
    
    for num,arrays in iterations:
        x,obj,der = unpackIterations(arrays)
        x_array.append(np.reshape(x,(nelx+1,nely+1),order='F'))
    
    SaveAsGif(x_array,nelx,nely,'GeneratedPart/FenicsOutput')
    
def saveStatsForEachIteration(iterationImages,formatVector):
    """
    Takes the sequence of images(or flattened fenics arrays) and tests the compliance and stress of each part.
    returns a dictionary of the stats over time.
    """
    
    complianceList = []
    stressList = []
    massList = []
    print("Testing each iteration")
    for part in iterationImages:
        if(part.ndim == 1):
            part_flat = part
        else:
            part_flat = np.ravel(part,order='F')

        #print("\tCompliance Max: {}".format(c_max))
        #print("\tStress Max: {}".format(s_max))

        compliance,stress = convergenceTester(formatVector,part_flat,1)
        mass = np.sum(part_flat)

        complianceList.append(float(compliance))
        stressList.append(float(stress))
        massList.append(float(mass))


    dictOfValues = {'mass':massList,'compliance':complianceList,'stress':stressList}

    return dictOfValues

def visualizeShiftDifferences(dataPoint):
    model = getModel(100,50)
    trueFormatVector,TruePart,converged = loadFenicsPart(dataPoint)
    fenicsAgentToGif(dataPoint)

    

    print(trueFormatVector)
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

    compliance,stress = convergenceTester(trueFormatVector,TruePart,1)
    mass = np.sum(TruePart)
    f = open("GeneratedPart/ModelComparison.txt",'w')

    f.write("Circle 1: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(trueFormatVector[0][0][0],trueFormatVector[0][1][0],trueFormatVector[1][0]))
    f.write("Circle 2: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(trueFormatVector[0][0][1],trueFormatVector[0][1][1],trueFormatVector[1][1]))
    f.write("Circle 3: ( {:.2f}, {:.2f} ) radius = {:.3f}\n".format(trueFormatVector[0][0][2],trueFormatVector[0][1][2],trueFormatVector[1][2]))

    f.write("\nForce 1: ( {:.2e}, {:.2e} ) magnitude = {:.3e}\n".format(trueFormatVector[2][0][0],trueFormatVector[2][1][0],np.sqrt(trueFormatVector[2][0][0]**2 + trueFormatVector[2][1][0]**2)))
    f.write("Force 2: ( {:.2e}, {:.2e} ) magnitude = {:.3e}\n".format(trueFormatVector[2][0][1],trueFormatVector[2][1][1],np.sqrt(trueFormatVector[2][0][1]**2 + trueFormatVector[2][1][1]**2)))
    f.write("Force 3: ( {:.2e}, {:.2e} ) magnitude = {:.3e}\n".format(trueFormatVector[2][0][2],trueFormatVector[2][1][2],np.sqrt(trueFormatVector[2][0][2]**2 + trueFormatVector[2][1][2]**2)))


    f.write("\nFenics part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))

    compliance,stress = convergenceTester(trueFormatVector,part_flat,1)
    mass = np.sum(part_flat)
    f.write("\nModel part:\n")
    f.write("\tmass: {}\n".format(int(mass)))
    f.write("\tCompliance: {:.5f}\n".format(compliance))
    f.write("\tStress: {:.5e}\n".format(stress))
    f.close()


    return actualImages[sortedScoreIndexes[0]]
    
    
    # fig,ax = plt.subplots(2*shiftRadius + 1,2*shiftRadius+1)

    
        
    
    # for i in range(2*shiftRadius+1):
    #     for j in range(2*shiftRadius+1):
    #         index = (2*shiftRadius+1)*i + j
    #         if(i==shiftRadius and j==shiftRadius):
    #             ax[i,j].imshow(truePart.T,cmap='gray_r',norm=colors.Normalize(0,1))
    #             ax[i,j].set_title("True")
    #         else:
    #             ax[i,j].imshow(actualImages[index].T,cmap='gray_r',norm=colors.Normalize(0,1))
    #             ax[i,j].set_title("{}:{:.3f}:({},{})".format(index,scores[index],shiftIndexes[index][1],shiftIndexes[index][0]))
    #         ax[i,j].get_xaxis().set_visible(False)
    #         ax[i,j].get_yaxis().set_visible(False)

        
    # plt.show()
        #print("File {}(converged={}) has difference {}".format(fileName,converged,diff))



if(__name__ == "__main__"):
    #path = r'E:\TopoptGAfileSaves\Mass minimization\AlienWareData\True\100_50_Validation'
    path = os.path.join(os.getcwd(),'Data','100_50')
    dataPoints = os.listdir(path)
    i = np.random.randint(0,len(dataPoints)-1)
    print("\n(",i,")\n")
    #scoreValidations(path,200)
    print(dataPoints[i])
    part = visualizeShiftDifferences(os.path.join(path,dataPoints[i]))
    #np.savetxt("out",part)

    # trueFormatVector,TruePart,converged = loadFenicPart(os.path.join(path,dataPoints[i]))
    # part_pred = np.ones((101,51))
    # part_true = np.reshape(TruePart,(101,51),order='F')

    # score = scoreModelPredictions(trueFormatVector,[part_pred,part_true])
    print("\n",i,"\n")
    #saveAsPVD(np.ravel(part,order='F'),100,50)

 