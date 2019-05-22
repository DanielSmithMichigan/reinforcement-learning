import tensorflow as tf
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
            name,
            numStateVariables,
            numActions,
            networkSize,
            entropyCoefficient,
            learningRate,
            maxGradientNorm,
            batchSize,
            weightRegularizationConstant,
            showGraphs
        ):
        self.sess = sess
        self.name = name
        self.numStateVariables = numStateVariables
        self.numActions = numActions
        self.networkSize = networkSize
        self.entropyCoefficient = entropyCoefficient
        self.learningRate = learningRate
        self.maxGradientNorm = maxGradientNorm
        self.batchSize = batchSize
        self.regTermOverTime = deque([], 100)
        self.entropyOverTime = deque([], 100)
        self.actionsChosenOverTime = []
        self.meanOverTime = deque([], 100)
        self.varianceOverTime = deque([], 100)
        self.entropyCostOverTime = deque([], 100)
        self.recentEntropy = deque([], 100)
        self.qCostOverTime = deque([], 100)
        self.actionOutputOverTime = deque([], 100)
        self.regLossOverTime = deque([], 100)
        self.weightRegularizationConstant = weightRegularizationConstant
        self.buildNetwork()
        if showGraphs:
            self.buildGraphs()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name=self.name + "_state")
            prevLayer = self.statePh
            layerNum = 0
            for i in self.networkSize:
                prevLayer = tf.layers.dense(inputs=prevLayer, units=i, activation=tf.nn.leaky_relu, name=self.name + "_dense_" + str(layerNum))
                layerNum = layerNum + 1
            self.actionMean = tf.layers.dense(inputs=prevLayer, units=self.numActions, name=self.name+"_actionMean")
            self.uncleanedActionVariance = tf.layers.dense(inputs=prevLayer, units=self.numActions, name=self.name+"_logScaleActionVariance", activation=tf.nn.tanh)
            self.logScaleActionVariance = -20 + 0.5 * (2 - -20) * (self.uncleanedActionVariance + 1)
            self.actionVariance = tf.exp(self.logScaleActionVariance)
            self.randoms = tf.random.normal(shape=tf.shape(self.actionMean), dtype=tf.float32)
            self.rawAction = self.actionMean + (self.randoms * self.actionVariance)
            self.actionsChosen = tf.tanh(self.rawAction)
            self.entropy = tf.reduce_sum(self.logScaleActionVariance + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
            self.logProb = gaussian_likelihood(self.rawAction, self.actionMean, self.logScaleActionVariance) - tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - self.actionsChosen ** 2, lower=0, upper=1) + EPS), axis=1)
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildGraphs(self):
        plt.ion()
        self.overview = plt.figure()
        self.overview.suptitle(self.name)
        self.regTermGraph = self.overview.add_subplot(4, 1, 1)
        self.entropyGraph = self.overview.add_subplot(4, 1, 2)
        self.actionChoicesGraph = self.overview.add_subplot(4, 1, 3)
        self.costOverTimeGraph = self.overview.add_subplot(4, 1, 4)
    def updateGraphs(self):
        self.regTermGraph.cla()
        self.regTermGraph.set_title("Reg Term")
        self.regTermGraph.plot(self.regTermOverTime)

        self.entropyGraph.cla()
        self.entropyGraph.set_title("Entropy")
        self.entropyGraph.plot(self.entropyOverTime)

        self.actionChoicesGraph.cla()
        self.actionChoicesGraph.set_title("Action Choices")
        self.actionChoicesGraph.plot(util.getColumn(self.meanOverTime, 0), label="mean")
        self.actionChoicesGraph.plot(util.getColumn(self.varianceOverTime, 0), label="variance")
        self.actionChoicesGraph.legend(loc=2)

        self.costOverTimeGraph.cla()
        self.costOverTimeGraph.set_title("Cost")
        self.costOverTimeGraph.plot(self.entropyCostOverTime, label="Entropy")
        self.costOverTimeGraph.plot(self.qCostOverTime, label="Q Value")
        self.costOverTimeGraph.plot(self.regLossOverTime, label="Reg Loss")
        self.costOverTimeGraph.legend(loc=2)

        self.overview.canvas.draw()
    def setQNetwork(self, qNetwork):
        self.qNetwork = qNetwork
    def buildTrainingOperation(self):
        self.entropyCoefficientPh = tf.placeholder(tf.float32, shape=1, name=self.name + "_entropyCoefficient")
        self.entropyLoss = self.entropyCoefficientPh * tf.reduce_mean(self.logProb)
        regLoss = self.weightRegularizationConstant * 0.5 * tf.reduce_mean(self.uncleanedActionVariance ** 2)
        regLoss += self.weightRegularizationConstant * 0.5 * tf.reduce_mean(self.actionMean ** 2)
        self.regLoss = regLoss
        self.qCost = -tf.reduce_mean(self.qNetwork.qValue)
        self.totalLoss = self.entropyLoss + self.qCost + self.regLoss
        self.loss = tf.reduce_mean(self.totalLoss)
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        self.uncappedGradients, variables = zip(*self.optimizer.compute_gradients(self.loss,var_list=self.networkParams))
        (
            self.cappedGradients,
            self.gradientNorm
        ) = tf.clip_by_global_norm(self.uncappedGradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.cappedGradients, variables))
    def trainAgainst(self, memories):
        (
            _,
            qCost,
            qValue,
            actionsChosen,
            rawAction,
            logProb,
            mean,
            variance,
            totalLoss,
            entropyLoss,
            loss,
            gradients,
            gradientNorm,
            regLoss
        ) = self.sess.run([
            self.trainingOperation,
            self.qCost,
            self.qNetwork.qValue,
            self.actionsChosen,
            self.rawAction,
            self.logProb,
            self.actionMean,
            self.logScaleActionVariance,
            self.totalLoss,
            self.entropyLoss,
            self.loss,
            self.cappedGradients,
            self.gradientNorm,
            self.regLoss
        ], feed_dict={
            self.statePh: util.getColumn(memories, constants.STATE),
            self.qNetwork.statePh: util.getColumn(memories, constants.STATE),
            self.entropyCoefficientPh: [self.entropyCoefficient]
        })

        # print("Gradients")
        # print(gradients)actionsPh
        # print("Gradient Norm")
        # print(gradientNorm)
        # print("ENTROPY LOSS: "+str(entropyLoss))
        # print("Q LOSS: "+str(qCost))
        self.qCostOverTime.append(np.mean(qCost))
        self.entropyCostOverTime.append(np.mean(entropyLoss))
        self.recentEntropy.append(np.mean(logProb))
        self.regTermOverTime.append(gradientNorm)
        self.regLossOverTime.append(regLoss)
        self.entropyOverTime.append(np.mean(logProb))
    def getAction(self, state):
        (
            rawAction,
            output,
            logProb,
            actionMean,
            logScaleActionVariance,
            entropy
        ) = self.sess.run([
            self.rawAction,
            self.actionsChosen,
            self.logProb,
            self.actionMean,
            self.logScaleActionVariance,
            self.entropy
        ], feed_dict={
            self.statePh: [state]
        })
        # print("MEAN: ",actionMean[0][0]," VARIANCE: ",logScaleActionVariance[0][0]," RAW: ",rawAction[0][0]," ENTROPY: ",entropy)
        self.actionsChosenOverTime.append(rawAction[0])
        self.meanOverTime.append(actionMean[0])
        self.varianceOverTime.append(np.exp(logScaleActionVariance[0]))
        self.actionOutputOverTime.append(output[0])
        return output[0], logScaleActionVariance[0], logProb[0], entropy

