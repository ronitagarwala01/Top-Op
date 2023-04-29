
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import optimizers
from keras.models import Sequential

iterationShape = (51, 101, 1)
unpackedShape = (20,)

'''
    Discriminator Models
'''
def discriminator2(iterationShape, unpackedShape):

    label_input = layers.Input(shape=(unpackedShape[0]))

    # CNN param to resize to
    nelx, nely = iterationShape[0], iterationShape[1]
    shape_0, shape_1 = nelx, nely
    n_nodes = shape_0 * shape_1

    # Label input
    labels = layers.Dense(n_nodes, name='Label-Input')(label_input)
    labels = layers.Reshape((shape_0, shape_1, 1), name='Label-Reshape')(labels)

    # This is our input from the image
    image_input = layers.Input(shape=(iterationShape), name='Image-Input')





    merge = layers.Concatenate()([labels, image_input])

    # These next few upscale layers are here so when we resize to 128x64,
    # (ideally) more information is preserved and not lost through the resizing operation
    # Layer 1
    upscale = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-1')(merge)
    upscale = layers.ReLU(name='Upscale-1-ReLU')(upscale)

    upscale = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-2')(upscale)
    upscale = layers.ReLU(name='Upscale-2-ReLU')(upscale)


    resize = layers.Resizing(height=128, width=64)(upscale)

    # Layer 2
    dis = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv1')(resize)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-1')(dis)

    # Layer 3
    dis = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv2')(dis)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-2')(dis)

    # Layer 4
    dis = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', name='conv3')(dis)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-3')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-1')(dis)


    dis = layers.Flatten()(dis)
    dis = layers.Dropout(0.2)(dis)


    output = layers.Dense(1, activation='sigmoid')(dis)
    # output = layers.Softmax()(output)


    model = keras.models.Model([image_input, label_input], output)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  metrics=['accuracy'])
    
    return model

# dis2 = discriminator2(iterationShape, unpackedShape)
# dis2.summary()





def discriminator3(iterationShape, unpackedShape):

    label_input = layers.Input(shape=(unpackedShape[0]))

    # CNN param to resize to
    nelx, nely = iterationShape[0], iterationShape[1]
    shape_0, shape_1 = nelx, nely
    n_nodes = shape_0 * shape_1

    # Label input
    labels = layers.Dense(n_nodes, name='Label-Input')(label_input)
    labels = layers.Reshape((shape_0, shape_1, 1), name='Label-Reshape')(labels)

    # This is our input from the image
    image_input = layers.Input(shape=(iterationShape), name='Image-Input')





    merge = layers.Concatenate()([labels, image_input])

    # These next few upscale layers are here so when we resize to 128x64,
    # (ideally) more information is preserved and not lost through the resizing operation
    # Layer 1
    upscale = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-1')(merge)
    upscale = layers.ReLU(name='Upscale-1-ReLU')(upscale)

    upscale = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-2')(upscale)
    upscale = layers.ReLU(name='Upscale-2-ReLU')(upscale)


    resize = layers.Resizing(height=128, width=64)(upscale)

    # Layer 2
    dis = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv1')(resize)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-1')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-1')(dis)

    # Layer 3
    dis = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv2')(dis)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-2')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-2')(dis)

    # Layer 4
    dis = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', name='conv3')(dis)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-3')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-3')(dis)


    dis = layers.Flatten()(dis)
    dis = layers.Dropout(0.2)(dis)


    output = layers.Dense(1, activation='sigmoid')(dis)
    # output = layers.Softmax()(output)


    model = keras.models.Model([image_input, label_input], output)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  metrics=['accuracy'])
    
    return model

# dis3 = discriminator3(iterationShape, unpackedShape)
# dis3.summary()




def discriminator4(iterationShape, unpackedShape):

    label_input = layers.Input(shape=(unpackedShape[0]))

    # CNN param to resize to
    nelx, nely = iterationShape[0], iterationShape[1]
    shape_0, shape_1 = nelx, nely
    n_nodes = shape_0 * shape_1

    # Label input
    labels = layers.Dense(n_nodes, name='Label-Input')(label_input)
    labels = layers.Reshape((shape_0, shape_1, 1), name='Label-Reshape')(labels)

    # This is our input from the image
    image_input = layers.Input(shape=(iterationShape), name='Image-Input')





    merge = layers.Concatenate()([labels, image_input])

    # These next few upscale layers are here so when we resize to 128x64,
    # (ideally) more information is preserved and not lost through the resizing operation
    # Layer 1
    upscale = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-1')(merge)
    upscale = layers.ReLU(name='Upscale-1-ReLU')(upscale)

    upscale = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-2')(upscale)
    upscale = layers.ReLU(name='Upscale-2-ReLU')(upscale)


    resize = layers.Resizing(height=128, width=64)(upscale)

    # Layer 2
    dis = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv1')(resize)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-1')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-1')(dis)



    # Layer 3
    dis = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv2')(dis)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-2')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-2')(dis)

    dis = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same', name='Upscale-2')(dis)
    dis = layers.ReLU(name='Upscale-1-ReLU')(dis)

    # Layer 4
    dis = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', name='conv3')(dis)
    dis = layers.LeakyReLU(alpha=0.2, name='ReLU-3')(dis)
    dis = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MaxPool-3')(dis)


    dis = layers.Flatten()(dis)
    dis = layers.Dropout(0.2)(dis)


    output = layers.Dense(1, activation='sigmoid')(dis)
    # output = layers.Softmax()(output)


    model = keras.models.Model([image_input, label_input], output)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  metrics=['accuracy'])
    
    return model

dis4 = discriminator4(iterationShape, unpackedShape)
dis4.summary()