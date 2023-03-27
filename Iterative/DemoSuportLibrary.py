from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tensorflow as tf
import os



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

def scoreModelPrediction(formatted,part):
    """
    Take the formatted array and a part and return the current mass, stress, and compliance of the part.
    
    Uses fenics to generate the stress and compliance scores
    """

    c_max = formatted[6]
    s_max = formatted[7]

    stress,compliance = fenics_testPart(formatted,part)
    stress = np.max(stress)

    mass = np.sum(np.ravel(part))

    score = mass + np.exp(compliance - c_max) + np.exp(stress)

    return score

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
        imageArray = []
        for i,image in enumerate(images):
            print("{:.2f}%\t".format(100*(i/len(images))))
            fig,ax = plt.subplots(1,1)
            
            if(i == 0):
                ax.set_title("Iteration: {}".format(i))
            else:
                im1 = np.reshape(image,((nelx+1)*(nely+1)))
                im2 = np.reshape(images[i-1],((nelx+1)*(nely+1)))
                ax.set_title("Iteration: {}, Change: {:.5f}".format(i,np.linalg.norm(im1-im2,ord=np.inf)))
            ax.imshow(np.reshape(image,(nelx+1,nely+1)).T,cmap='gray_r',norm=colors.Normalize(vmin=0,vmax=1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')

            im = Image.open(img_buf)
            imageArray.append(im)

            #plt.show()
        im = imageArray[0]
        imageArray.pop(0)
        im.save(str(name) + ".gif".format(i),save_all=True,append_images = imageArray,optimize=False,loop=0)
        im.close()
    
def loadFenicPart(agentFileToGet):
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
    
    print(markName)
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




