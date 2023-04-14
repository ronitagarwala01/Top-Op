import tensorflow as tf
import numpy as np

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
    #conv_64_32 = tf.keras.layers.Dropout(rate=0.1)(conv_64_32)

    #Second convolution Layer
    conv_64_32 = tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3),padding='same',activation=activation)(conv_64_32)
    conv_64_32 = tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3),padding='same',activation=activation)(conv_64_32)
    conv_32_16 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv_64_32)
    conv_32_16 = tf.keras.layers.GaussianNoise(stddev=0.1)(conv_32_16)
    #conv_32_16 = tf.keras.layers.Dropout(rate=0.1)(conv_32_16)

    conv_32_16 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3),padding='same',activation=activation)(conv_32_16)
    conv_32_16 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3),padding='same',activation=activation)(conv_32_16)
    conv_16_8 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(conv_32_16)
    conv_16_8 = tf.keras.layers.GaussianNoise(stddev=0.1)(conv_16_8)
    #conv_16_8 = tf.keras.layers.Dropout(rate=0.1)(conv_16_8)

    conv_16_8 = tf.keras.layers.Conv2D(filters= 128, kernel_size=(3,3),padding='same',activation=activation)(conv_16_8)
    conv_16_8 = tf.keras.layers.Conv2D(filters= 128, kernel_size=(3,3),padding='same',activation=activation)(conv_16_8)

    #upscaleLayer
    #upscaling is performed by convolution transpose where stride=2 < kernalsize
    convUpscale_32_16 = tf.keras.layers.Conv2DTranspose(filters= 64, kernel_size=(5,5),strides=2,padding='same',activation=activation)(conv_16_8)
    convUpscale_32_16 = tf.keras.layers.GaussianNoise(stddev=0.1)(convUpscale_32_16)
    convUpscale_32_16 = ConcatAndCrop(convUpscale_32_16.shape,conv_32_16.shape)(convUpscale_32_16,conv_32_16)
    convUpscale_32_16 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_32_16)
    convUpscale_32_16 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_32_16)
    #convUpscale_32_16 = tf.keras.layers.Dropout(rate=0.1)(convUpscale_32_16)

    convUpscale_64_32 = tf.keras.layers.Conv2DTranspose(filters= 32, kernel_size=(5,5),strides=2,padding='same',activation=activation)(convUpscale_32_16)
    convUpscale_64_32 = tf.keras.layers.GaussianNoise(stddev=0.1)(convUpscale_64_32)
    convUpscale_64_32 = ConcatAndCrop(convUpscale_64_32.shape,conv_64_32.shape)(convUpscale_64_32,conv_64_32)
    convUpscale_64_32 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_64_32)
    convUpscale_64_32 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3),strides=1,padding='same',activation=activation)(convUpscale_64_32)
    #convUpscale_64_32 = tf.keras.layers.Dropout(rate=0.1)(convUpscale_64_32)

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


class TopOptSequence:
    """
    Class to hold the mass minimization sequences. holds the load conditions and all iterations.
    Inlcudes functions to convert data to model ready inputs.
    """
    def __init__(self,ID,formatted,x_array,numIterations,converged):
        self.ID = ID
        self.loadCondtions = formatted
        self.xPhys_array = x_array
        self.numIterations = numIterations
        self.nelx = self.loadCondtions[3]
        self.nely = self.loadCondtions[4]
        self.converged = converged
    
    def formatLoadCondtions(self,AugmentConditions:bool = False):
        """
        Generates the loadcondtions image from the current sequence load conditions.

        parameters:
            - AugmentConditions: boolean to flag if the part parameters(ComplianceMax and StressMax) should be randomly augmented to a higher value.
                - This will help the model learn that the min stress and compliance does not need to be reached.
        
        returns:
            - loadConditions: an nd.array with shape (nelx+1,nely+1,6) holding the load condition image
        """
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
        #circleImage = np.where(np.abs(circleImage) <= 0.01 , 1,0)

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

        if(AugmentConditions):
            #increase the compliance and stress max of the given part by up to 50%
            if(np.random.random() >= 0.5):
                c_max_image = c_max_image * (np.random.random()*0.5 + 1)
            else:
                s_max_image = s_max_image * (np.random.random()*0.5 + 1)

        # print("circleImage.shape:",circleImage.shape)
        # print("forceImageX.shape:",forceImageX.shape)
        # print("forceImageY.shape:",forceImageY.shape)
        # print("Y_image.shape:",Y_image.shape)
        # print("c_max_image.shape:",c_max_image.shape)
        # print("s_max_image.shape:",s_max_image.shape)

        loadCondtionsImage = np.concatenate([circleImage,forceImageX,forceImageY,Y_image,c_max_image,s_max_image],axis=2)
        loadCondtionsImage = np.reshape(loadCondtionsImage,(nelx+1,nely+1,6))
        return loadCondtionsImage

    def dispenceIteration(self,iterationNumber,iterationdepth:int=5,step:int=5,augmentData:bool = False):
        iterationNumber = min(iterationNumber,self.numIterations-1)
        StartingBlock = np.reshape(self.xPhys_array[iterationNumber],(self.nelx+1,self.nely+1,1),order='F')
        outputParts = []
        for i in range(iterationdepth):
            
            jumpIndex = min(self.numIterations-1,iterationNumber + step*(i+1))
            outputParts.append(np.reshape(self.xPhys_array[jumpIndex],(self.nelx+1,self.nely+1,1),order='F'))

        
        formattedImage = self.formatLoadCondtions(augmentData)

        return StartingBlock,formattedImage,outputParts

    def dispenceFullPartIteraion(self,iterationdepth:int=5,augmentData:bool = False):
        """
        Dispence the starting block as the first iteration and keep the load conditions the same, but set the ouput parts as every major iteration from start to fininsh that fits into iterationDepth

        Parameters:
            - iterationDepth: number of output files to produce, should match up with the number of outputs the model has
                - doubles as the inverse step size for selecting images, now each part will be indexed as (numIteration/iterationDepth) iterations appart.
            - AugmentData: bool to decide if load conditions should be slightly increased when building loax condtions image.
        """
        StartingBlock = np.reshape(self.xPhys_array[0],(self.nelx+1,self.nely+1,1),order='F')

        
        start = self.numIterations//iterationdepth
        steps = np.linspace(start,self.numIterations,iterationdepth,endpoint=True,dtype='int32')
        outputParts = []
        for i in steps:
            
            jumpIndex = min(self.numIterations-1, i)
            outputParts.append(np.reshape(self.xPhys_array[jumpIndex],(self.nelx+1,self.nely+1,1),order='F'))

        
        formattedImage = self.formatLoadCondtions(augmentData)

        return StartingBlock,formattedImage,outputParts



