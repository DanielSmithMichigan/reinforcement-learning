import tensorflow as tf
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import time

from . import constants
from . import util
from collections import deque

EPS=float(1e-6)

def gaussian_likelihood(input_, mu_, log_std):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.
    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


class PolicyNetwork:
    def __init__(self,
            sess,
            graph,
            name,
            numStateVariables,
            numActions,
            networkSize,
            entropyCoefficient,
            learningRate,
            maxGradientNorm,
            batchSize,
            meanRegularizationConstant,
            varianceRegularizationConstant,
            showGraphs,
            statePh
        ):
        self.sess = sess
        self.graph = graph
        self.name = name
        self.numStateVariables = numStateVariables
        self.numActions = numActions
        self.networkSize = networkSize
        self.entropyCoefficient = entropyCoefficient
        self.learningRate = learningRate
        self.maxGradientNorm = maxGradientNorm
        self.batchSize = batchSize
        self.statePh = statePh
        self.meanRegularizationConstant = meanRegularizationConstant
        self.varianceRegularizationConstant = varianceRegularizationConstant
        with self.graph.as_default():
            self.entropyCoefficientPh = tf.placeholder(
                tf.float32,
                shape=[],
                name=self.name + "_entropyCoefficient"
            )
    def buildNetwork(self, state):
        actionsChosen = None
        rawAction = None
        actionMean = None
        uncleanedActionVariance = None
        logScaleActionVariance = None
        actionVariance = None
        entropy = None
        logProb = None
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
                actionMean = tf.layers.dense(
                    inputs=prevLayer,
                    units=self.numActions,
                    name="actionMean",
                    reuse=tf.AUTO_REUSE
                )
                uncleanedActionVariance = tf.layers.dense(
                    inputs=prevLayer,
                    units=self.numActions,
                    name="logScaleActionVariance",
                    reuse=tf.AUTO_REUSE,
                    activation=tf.nn.tanh
                )
                logScaleActionVariance = -20 + 0.5 * (2 - -20) * (uncleanedActionVariance + 1)
                actionVariance = tf.exp(logScaleActionVariance)
                randoms = tf.random.normal(shape=tf.shape(actionMean), dtype=tf.float32)
                rawAction = actionMean + (randoms * actionVariance)
                actionsChosen = tf.tanh(rawAction)
                entropy = tf.reduce_sum(logScaleActionVariance + 0.5 * np.log(2.0 * np.pi * np.e), axis=1)
                logProb = gaussian_likelihood(rawAction, actionMean, logScaleActionVariance) - tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - actionsChosen ** 2, lower=0, upper=1) + EPS), axis=1)
        return (
            actionsChosen,
            rawAction,
            actionMean,
            uncleanedActionVariance,
            logScaleActionVariance,
            actionVariance,
            entropy,
            logProb
        )
    def setQNetwork(self, qNetwork):
        self.qNetwork = qNetwork
    def buildTrainingOperation(self):
        with self.graph.as_default():
            (
                actionsChosen,
                _,
                actionMean,
                uncleanedActionVariance,
                _,
                _,
                _,
                logProb
            ) = self.buildNetwork(self.statePh)
            entropyLoss = self.entropyCoefficientPh * tf.reduce_mean(logProb)
            regLoss = 0 \
                + self.varianceRegularizationConstant * 0.5 * tf.reduce_mean(uncleanedActionVariance ** 2) \
                + self.meanRegularizationConstant * 0.5 * tf.reduce_mean(actionMean ** 2)
            qCost = -tf.reduce_mean(
                self.qNetwork.buildNetwork(
                    self.statePh,
                    actionsChosen
                )
            )
            totalLoss = entropyLoss + qCost + regLoss
            loss = tf.reduce_mean(totalLoss)
            optimizer = tf.train.AdamOptimizer(self.learningRate)
            uncappedGradients, variables = zip(
                *optimizer.compute_gradients(
                    loss,
                    var_list=tf.trainable_variables(scope=self.name)
                )
            )
            (
                cappedGradients,
                self.gradientNorm
            ) = tf.clip_by_global_norm(uncappedGradients, self.maxGradientNorm)
            return optimizer.apply_gradients(zip(cappedGradients, variables))

