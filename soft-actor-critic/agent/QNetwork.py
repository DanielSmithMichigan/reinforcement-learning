import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import util
from . import constants

class QNetwork:
    def __init__(self,
            sess,
            graph,
            name,
            numStateVariables,
            numActions,
            networkSize,
            gamma,
            learningRate,
            showGraphs,
            statePh,
            nextStatePh,
            actionsPh,
            rewardsPh,
            terminalsPh,
            maxGradientNorm
        ):
        self.sess = sess
        self.graph = graph
        self.statePh = statePh
        self.nextStatePh = nextStatePh
        self.actionsPh = actionsPh
        self.rewardsPh = rewardsPh
        self.terminalsPh = terminalsPh
        self.maxGradientNorm = maxGradientNorm
        self.name = name
        self.numStateVariables = numStateVariables
        self.numActions = numActions
        self.networkSize = networkSize
        self.gamma = gamma
        self.learningRate = learningRate
        self.storedTargets = None
    def buildNetwork(self, state, actions):
        qValue = None
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                prevLayer = tf.concat(
                    [
                        state,
                        actions
                    ],
                    axis=-1
                )
                for i in range(len(self.networkSize)):
                    prevLayer = tf.layers.dense(
                        inputs=prevLayer,
                        units=self.networkSize[i],
                        activation=tf.nn.relu,
                        name="dense_" + str(i),
                        reuse=tf.AUTO_REUSE
                    )
                qValue = tf.layers.dense(
                    inputs=prevLayer,
                    units=1,
                    name="qValue",
                    reuse=tf.AUTO_REUSE
                )
        return qValue
    def setValueNetwork(self, valueNetwork):
        self.valueNetwork = valueNetwork
    def setPolicyNetwork(self, policyNetwork):
        self.policyNetwork = policyNetwork
    def buildTrainingOperation(self):
        with self.graph.as_default():
            reshapedValue = tf.reshape(self.valueNetwork.buildNetwork(self.nextStatePh), [-1])
            targetQ = self.rewardsPh + self.gamma * (1 - self.terminalsPh) * tf.stop_gradient(reshapedValue)
            predictedQ = self.buildNetwork(self.statePh, self.actionsPh)
            predictedQ = tf.reshape(predictedQ, [-1])
            absDiff = targetQ - predictedQ
            loss = 0.5 * tf.reduce_mean(absDiff ** 2)
            optimizer = tf.train.AdamOptimizer(self.learningRate)
            uncappedGradients, variables = zip(
                *optimizer.compute_gradients(
                    loss,
                    var_list=tf.trainable_variables(scope=self.name)
                )
            )
            (
                cappedGradients,
                regTerm
            ) = tf.clip_by_global_norm(uncappedGradients, self.maxGradientNorm)
            return optimizer.apply_gradients(zip(cappedGradients, variables)), loss, regTerm
    def buildAssessmentOperation(self, actions):
        with self.graph.as_default():
            self.assessmentOperation = self.buildNetwork(self.statePh, actions)

