import tensorflow as tf




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





