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

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -20


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


def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal gaussian distribution.

    :param log_std: (tf.Tensor) Log of the standard deviation
    :return: (tf.Tensor)
    """
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the ouput of the gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi


class PolicyNetwork:
    def __init__(self,
            sess,
            graph,
            name,
            numStateVariables,
            numActions,
            networkSize,
            learningRate,
            batchSize,
            showGraphs,
            statePh,
            targetEntropy
        ):
        self.sess = sess
        self.graph = graph
        self.name = name
        self.numStateVariables = numStateVariables
        self.numActions = numActions
        self.targetEntropy = targetEntropy
        self.networkSize = networkSize
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.statePh = statePh
        with self.graph.as_default():
            with tf.variable_scope("EntropyCoefficient"):
                self.logEntropyCoefficient = tf.get_variable(
                    'logEntropyCoefficient',
                    dtype=tf.float32,
                    initializer=np.log(1.0).astype(np.float32)
                )
                self.entropyCoefficient = tf.exp(self.logEntropyCoefficient)
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
                    reuse=tf.AUTO_REUSE
                )
                logScaleActionVariance = tf.clip_by_value(uncleanedActionVariance, LOG_STD_MIN, LOG_STD_MAX)
                actionVariance = tf.exp(logScaleActionVariance)
                randoms = tf.random.normal(tf.shape(actionMean))
                rawAction = actionMean + (randoms * actionVariance)
                logProb = gaussian_likelihood(rawAction, actionMean, logScaleActionVariance)
                entropy = gaussian_entropy(logScaleActionVariance)
                deterministicActionChosen, actionsChosen, logProb = apply_squashing_func(actionMean, rawAction, logProb)
        return (
            actionsChosen,
            rawAction,
            actionMean,
            uncleanedActionVariance,
            logScaleActionVariance,
            actionVariance,
            entropy,
            logProb,
            deterministicActionChosen
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
                logProb,
                _
            ) = self.buildNetwork(self.statePh)

            #Entropy Coefficient
            entropyCoefficientLoss = -tf.reduce_mean(
                self.logEntropyCoefficient * tf.stop_gradient(logProb + self.targetEntropy)
            )
            entropyOptimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
            entropyCoefficientTrainingOperation = entropyOptimizer.minimize(entropyCoefficientLoss, var_list=self.logEntropyCoefficient)

            #Policy
            qValue = self.qNetwork.buildNetwork(self.statePh, actionsChosen)
            qValue = tf.reshape(qValue, [-1])
            batchLoss = self.entropyCoefficient * logProb - qValue
            loss = tf.reduce_mean(
                batchLoss
            )
            # loss = tf.reduce_mean(
            #     tf.stop_gradient(self.entropyCoefficient) * logProb - self.qNetwork.buildNetwork(self.statePh, actionsChosen)
            # )
            optimizer = tf.train.AdamOptimizer(self.learningRate)
            policyTrainingOperation = optimizer.minimize(loss, var_list=tf.trainable_variables(scope=self.name))
            return policyTrainingOperation, entropyCoefficientTrainingOperation

