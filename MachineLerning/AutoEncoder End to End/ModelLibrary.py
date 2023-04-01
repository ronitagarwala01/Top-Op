import tensorflow as tf
import numpy as np


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self):
        super(Sampling,self).__init__()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim),mean=0,stddev=.1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def buildEncoder(inputshape,outputDims,denseUnits:int=10,activation='relu'):
    inputLayer = tf.keras.layers.Input(shape=inputshape,name="Input")

    reshapeLayer = tf.keras.layers.Flatten()(inputLayer)

    #start with a simple few dense layers for a simple input space
    denseLayer = tf.keras.layers.Dense(denseUnits,activation=activation)(reshapeLayer)
    denseLayer = tf.keras.layers.Dense(denseUnits,activation=activation)(denseLayer)

    z_mean = tf.keras.layers.Dense(outputDims,activation=activation,name="z_mean")(denseLayer)
    z_log_var = tf.keras.layers.Dense(outputDims,activation=activation,name="z_log_var")(denseLayer)

    z = Sampling()([z_mean,z_log_var])
    return tf.keras.Model(inputs = inputLayer, outputs = [z_mean, z_log_var, z], name="encoder")

def buildDecoder(latenDims,outputShape,denseUnits:int = 10, activation='relu'):
    inputLayer = tf.keras.Input(shape=(latenDims))

    denseLayer = tf.keras.layers.Dense(denseUnits, activation=activation)(inputLayer)
    denseLayer = tf.keras.layers.Dense(denseUnits, activation=activation)(denseLayer)
    denseLayer = tf.keras.layers.Dense(np.prod(outputShape),activation = "hard_sigmoid")(inputLayer)

    outputLayer = tf.keras.layers.Reshape(outputShape)(denseLayer)
    return tf.keras.Model(inputs = inputLayer, outputs = outputLayer, name="decoder")
