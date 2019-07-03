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
            memoryPriorityPh,
            maxGradientNorm
        ):
        self.sess = sess
        self.graph = graph
        self.statePh = statePh
        self.nextStatePh = nextStatePh
        self.actionsPh = actionsPh
        self.rewardsPh = rewardsPh
        self.terminalsPh = terminalsPh
        self.memoryPriorityPh = memoryPriorityPh
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
    def setTargetNetworks(self, target1, target2):
        self.target1 = target1
        self.target2 = target2
    def setPolicyNetwork(self, policyNetwork):
        self.policyNetwork = policyNetwork
    def buildTrainingOperation(self):
        with self.graph.as_default():
            (
                nextActionsChosen,
                _,
                _,
                _,
                _,
                _,
                _,
                nextLogProb,
                _
            ) = self.policyNetwork.buildNetwork(self.nextStatePh)
            nextQValuesQ1 = self.target1.buildNetwork(self.nextStatePh, nextActionsChosen)
            nextQValuesQ2 = self.target2.buildNetwork(self.nextStatePh, nextActionsChosen)
            minQ = tf.concat([nextQValuesQ1, nextQValuesQ2], axis=1)
            minQ = tf.reduce_min(minQ, axis=1)
            minQ = tf.reshape(minQ, [-1])
            entropyBonus = -1 * tf.stop_gradient(self.policyNetwork.entropyCoefficient) * nextLogProb
            targetQ = self.rewardsPh + self.gamma * (1 - self.terminalsPh) * (tf.stop_gradient(minQ) + entropyBonus)
            predictedQ = self.buildNetwork(self.statePh, self.actionsPh)
            predictedQ = tf.reshape(predictedQ, [-1])
            absDiff = targetQ - predictedQ
            batchwiseLoss = .5 * (absDiff ** 2)
            proportionedLoss = batchwiseLoss / self.memoryPriorityPh
            loss = tf.reduce_mean(proportionedLoss)
            # tf.summary.scalar(self.name+" Loss", loss)
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
            return optimizer.apply_gradients(zip(cappedGradients, variables)), loss, regTerm, batchwiseLoss
    def buildAssessmentOperation(self, actions):
        with self.graph.as_default():
            self.assessmentOperation = self.buildNetwork(self.statePh, actions)

