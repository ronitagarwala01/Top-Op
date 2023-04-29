import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import optimizers
from keras.models import Sequential

iterationShape = (51, 101, 1)
unpackedShape = (20,)


'''
    Generator Models
'''
def generator2(iterationShape, unpackedShape):
    # Label inputs of model, this will hold (roughly) true for every architecture
    label_input = layers.Input(shape=(unpackedShape[0]), name='Label-Input')
    

    # CNN param to resize to
    nelx, nely = iterationShape[0], iterationShape[1]
    shape_0, shape_1 = nelx, nely
    n_nodes = shape_0 * shape_1

    # Label input
    labels = layers.Dense(n_nodes, name='Label-Dense')(label_input)
    labels = layers.Reshape((shape_0, shape_1, 1), name='Label-Reshape')(labels)

    # Update to n_nodes to match Gen branch
    n_nodes *= unpackedShape[0]

    # Generator noise array input model
    noise_input = layers.Input(shape=iterationShape, name='Noise-Input')


    merge = layers.Concatenate()([noise_input, labels])
    resize = layers.Resizing(height=128, width=64)(merge)




    # Layer 1
    gen = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same')(resize)
    gen = layers.ReLU()(gen)

    # Layer 2
    gen = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.MaxPool2D(pool_size=(2,2))(gen)

    #  Layer 3
    gen = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(1,1),
                                padding='same')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.MaxPool2D(pool_size=(2,2))(gen)

    # Output & Model object
    output = layers.Conv2D(filters= 1, kernel_size=(7,7), activation='tanh',
                           padding='same')(gen)
    output = layers.Resizing(height=nelx, width=nely)(output)

    model = keras.models.Model([noise_input, label_input], output)
    
    return model

# gen2 = generator2(iterationShape, unpackedShape)
# gen2.summary()


def generator3(iterationShape, unpackedShape):
    # Label inputs of model, this will hold (roughly) true for every architecture
    label_input = layers.Input(shape=(unpackedShape[0]), name='Label-Input')
    

    # CNN param to resize to
    nelx, nely = iterationShape[0], iterationShape[1]
    shape_0, shape_1 = nelx, nely
    n_nodes = shape_0 * shape_1

    # Label input
    labels = layers.Dense(n_nodes, name='Label-Dense')(label_input)
    labels = layers.Reshape((shape_0, shape_1, 1), name='Label-Reshape')(labels)

    # Update to n_nodes to match Gen branch
    n_nodes *= unpackedShape[0]

    # Generator noise array input model
    noise_input = layers.Input(shape=iterationShape, name='Noise-Input')


    merge = layers.Concatenate()([noise_input, labels])
    resize = layers.Resizing(height=128, width=64)(merge)




    # Layer 1
    gen = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same')(resize)
    gen = layers.ReLU()(gen)

    # Layer 2
    gen = layers.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2),
                                padding='same')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.MaxPool2D(pool_size=(2,2))(gen)

    #  Layer 3
    gen = layers.Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(1,1),
                                padding='same')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.MaxPool2D(pool_size=(2,2))(gen)

    # Output & Model object
    output = layers.Conv2D(filters= 1, kernel_size=(7,7), activation='tanh',
                           padding='same')(gen)
    output = layers.Resizing(height=nelx, width=nely)(output)

    model = keras.models.Model([noise_input, label_input], output)
    
    return model

gen3 = generator3(iterationShape, unpackedShape)
gen3.summary()



def generator4(iterationShape, unpackedShape):
    # Label inputs of model, this will hold (roughly) true for every architecture
    label_input = layers.Input(shape=(unpackedShape[0]), name='Label-Input')
    

    # CNN param to resize to
    nelx, nely = iterationShape[0], iterationShape[1]
    shape_0, shape_1 = nelx, nely
    n_nodes = shape_0 * shape_1

    # Label input
    labels = layers.Dense(n_nodes, name='Label-Dense')(label_input)
    labels = layers.Reshape((shape_0, shape_1, 1), name='Label-Reshape')(labels)

    # Update to n_nodes to match Gen branch
    n_nodes *= unpackedShape[0]

    # Generator noise array input model
    noise_input = layers.Input(shape=iterationShape, name='Noise-Input')


    merge = layers.Concatenate()([noise_input, labels])
    resize = layers.Resizing(height=128, width=64)(merge)




    # Layer 1
    gen = layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2),
                                padding='same')(resize)
    gen = layers.ReLU()(gen)

    # Layer 2
    gen = layers.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2),
                                padding='same')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.MaxPool2D(pool_size=(2,2))(gen)

    #  Layer 3
    gen = layers.Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(1,1),
                                padding='same')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.MaxPool2D(pool_size=(2,2))(gen)

    # Output & Model object
    output = layers.Conv2D(filters= 1, kernel_size=(7,7), activation='tanh',
                           padding='same')(gen)
    output = layers.Resizing(height=nelx, width=nely)(output)

    model = keras.models.Model([noise_input, label_input], output)
    
    return model

gen4 = generator4(iterationShape, unpackedShape)
gen4.summary()


