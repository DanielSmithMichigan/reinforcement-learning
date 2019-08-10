import tensorflow as tf
import numpy as np
import math

def noisy_dense_layer(inputs, num_units, debug=False):
    initValue = 0.05
    fan_in = 1
    lastVal = None
    for i in inputs.get_shape().as_list():
        if i != None:
            fan_in = fan_in * i
        lastVal = i
    p = math.sqrt(3 / fan_in)
    kernelSize = [lastVal, num_units]
    kernelSizeTuple = (lastVal, num_units)

    weightNoise = tf.random_uniform(shape=kernelSize, minval=-1, maxval=1, dtype=tf.float32)
    weightNoiseCoefficient = tf.Variable(np.tile(initValue, kernelSizeTuple), dtype=tf.float32)
    weightNoiseOut = tf.multiply(weightNoise, weightNoiseCoefficient)

    actualWeightsOut = tf.Variable(np.random.uniform(low=-p, high=p, size=kernelSizeTuple), dtype=tf.float32)
    weightsOut = tf.add(weightNoiseOut, actualWeightsOut)
    output = tf.tensordot(inputs, weightsOut, 1)

    biasSize = [num_units]
    biasSizeTuple = [num_units]
    biasNoise = tf.random_uniform(shape=biasSize, minval=-1, maxval=1, dtype=tf.float32)
    biasNoiseCoefficient = tf.Variable(np.tile(initValue, biasSizeTuple), dtype=tf.float32)
    biasNoiseOut = tf.multiply(biasNoise, biasNoiseCoefficient)

    actualBiasesOut = tf.Variable(np.random.uniform(low=-p, high=p, size=biasSizeTuple), dtype=tf.float32)
    biasOut = tf.add(biasNoiseOut, actualBiasesOut)

    output = tf.nn.bias_add(output, biasOut)
    if debug:
        return output, weightNoiseOut, biasNoiseOut
    return output

