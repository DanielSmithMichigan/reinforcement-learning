import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import math
from collections import deque

from . import util
from . import constants

class ValueNetwork:
    def __init__(self,
            sess,
            graph,
            name,
            numStateVariables,
            networkSize,
            entropyCoefficient,
            learningRate,
            maxGradientNorm,
            batchSize,
            numActions,
            showGraphs,
            statePh
        ):
        self.sess = sess
        self.graph = graph
        self.name = name
        self.numStateVariables = numStateVariables
        self.networkSize = networkSize
        self.entropyCoefficient = entropyCoefficient
        self.learningRate = learningRate
        self.maxGradientNorm = maxGradientNorm
        self.batchSize = batchSize
        self.numActions = numActions
        self.statePh = statePh
    def buildNetwork(self, state):
        value = None
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                prevLayer = state
                for i in range(len(self.networkSize)):
                    prevLayer = tf.layers.dense(
                        inputs=prevLayer,
                        units=self.networkSize[i],
                        activation=tf.nn.relu,
                        name="dense_" + str(i),
                        reuse=tf.AUTO_REUSE
                    )
                value = tf.layers.dense(
                    inputs=prevLayer,
                    name="value",
                    units=1,
                    reuse=tf.AUTO_REUSE
                )
        return value
    def buildSoftCopyOperation(self, otherNetwork, tau):
        with self.graph.as_default():
            return [tf.assign(t, (1 - tau) * t + tau * e) for t, e in zip(
                tf.trainable_variables(
                    scope=self.name
                ),
                tf.trainable_variables(
                    scope=otherNetwork.name
                )
            )]
    def setNetworks(self, policyNetwork, qNetwork1, qNetwork2):
        self.policyNetwork = policyNetwork
        self.qNetwork1 = qNetwork1
        self.qNetwork2 = qNetwork2
    def buildTrainingOperation(self):
        with self.graph.as_default():
            (
                actionsChosen,
                _,
                _,
                _,
                _,
                _,
                _,
                logProb
            ) = self.policyNetwork.buildNetwork(self.statePh)
            minQValue = tf.minimum(
                self.qNetwork1.buildNetwork(self.statePh, actionsChosen),
                self.qNetwork2.buildNetwork(self.statePh, actionsChosen)
            )
            entropyValue = tf.reshape(self.entropyCoefficient * logProb, [-1, 1])
            targetValue = minQValue - entropyValue
            value = self.buildNetwork(self.statePh)
            loss = tf.reduce_mean(tf.pow(value - tf.stop_gradient(targetValue), 2))
            optimizer = tf.train.AdamOptimizer(self.learningRate)
            uncappedGradients, variables = zip(
                *optimizer.compute_gradients(
                    loss,
                    var_list=tf.trainable_variables(scope=self.name)
                )
            )
            (
                cappedGradients,
                gradientNorm
            ) = tf.clip_by_global_norm(uncappedGradients, self.maxGradientNorm)
            return optimizer.apply_gradients(zip(cappedGradients, variables))

