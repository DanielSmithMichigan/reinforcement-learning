import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

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
            numQuantiles,
            gamma,
            kappa,
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
        self.numQuantiles = numQuantiles
        self.gamma = gamma
        self.kappa = kappa
        self.learningRate = learningRate
        with self.graph.as_default():
            self.quantileThresholds = tf.constant(
                np.linspace(0, 1, numQuantiles + 2)[1:numQuantiles+1],
                dtype=tf.float32,
                shape=[self.numQuantiles]
            )
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
                        activation=tf.nn.leaky_relu,
                        name=self.name+"_dense_pre_" + str(i),
                        reuse=tf.AUTO_REUSE
                    )
                # batchSize x N
                quantileValues = tf.layers.dense(
                    inputs=prevLayer,
                    units=self.numQuantiles,
                    name=self.name+"_quantile_values",
                    reuse=tf.AUTO_REUSE
                )
                # batchSize x numQuantiles
                qValue = tf.reduce_mean(quantileValues, axis=1)
                # batchSize
        return quantileValues, qValue
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
    def setTargetNetworks(self, targetNetwork):
        self.targetNetwork = targetNetwork
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
            (
                nextQuantileValues,
                _
            ) = self.targetNetwork.buildNetwork(self.nextStatePh, nextActionsChosen)
            # batchSize x numQuantiles
            rewardsPh = tf.reshape(self.rewardsPh, [-1, 1])
            # batchSize x 1
            terminalsPh = tf.reshape(self.terminalsPh, [-1, 1])
            # batchSize x 1
            entropyBonus = tf.reshape(-1 * self.policyNetwork.entropyCoefficient * nextLogProb, [-1, 1])
            # batchSize x 1
            targetValues = rewardsPh + self.gamma * (1 - terminalsPh) * tf.stop_gradient(nextQuantileValues + entropyBonus)
            # batchSize x numQuantiles
            targetValues = tf.reshape(targetValues, [-1, 1, self.numQuantiles])
            # batchSize x 1 x numQuantiles
            targets = tf.tile(targetValues, [1, self.numQuantiles, 1])
            # Tiled down columns
            # batchSize x numQuantiles (tiled) x numQuantiles
            (
                predictedQuantileValues,
                _
            ) = self.buildNetwork(self.statePh, self.actionsPh)
            predictionValues = tf.reshape(predictedQuantileValues, [-1, self.numQuantiles, 1])
            # batchSize x numQuantiles x 1
            predictions = tf.tile(predictionValues, [1, 1, self.numQuantiles])
            # Tiled accross rows
            # batchSize x numQuantiles x numQuantiles (tiled)
            absDiff = tf.abs(targets - predictions)
            # batchSize x numQuantiles x numQuantiles
            minorError = tf.stop_gradient(tf.to_float(absDiff <= self.kappa)) * 0.5 * absDiff ** 2
            # batchSize x numQuantiles x numQuantiles
            majorError = tf.stop_gradient(tf.to_float(absDiff > self.kappa)) * self.kappa * (absDiff - 0.5 * self.kappa)
            # batchSize x numQuantiles x numQuantiles
            totalError = minorError + majorError
            # batchSize x numQuantiles x numQuantiles
            comparison = tf.stop_gradient(tf.to_float(targets < predictions))
            quantileThresholds = tf.reshape(self.quantileThresholds, [1, self.numQuantiles, 1])
            # numQuantiles
            quantilePunishment = tf.abs(quantileThresholds - comparison)
            # batchSize x numQuantiles x numQuantiles
            quantileRegressionLoss = quantilePunishment * totalError / self.kappa
            # batchSize x numQuantiles x numQuantiles
            perQuantileLoss = tf.reduce_sum(quantileRegressionLoss, axis=2)
            # batchSize x numQuantiles
            batchwiseLoss = tf.reduce_mean(perQuantileLoss, axis=1)
            # batchSize
            # proportionedLoss = batchwiseLoss / self.memoryPriorityPh
            loss = tf.reduce_mean(batchwiseLoss)
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
            optimizer = optimizer.apply_gradients(zip(cappedGradients, variables))
            return (
                optimizer,
                loss,
                regTerm,
                batchwiseLoss,
                nextLogProb,
                nextQuantileValues,
                entropyBonus,
                targetValues,
                targets,
                predictionValues,
                predictions,
                absDiff,
                minorError,
                majorError,
                totalError,
                quantilePunishment,
                quantileRegressionLoss,
                perQuantileLoss
            )
    def buildAssessmentOperation(self, actions):
        with self.graph.as_default():
            self.assessmentOperation = self.buildNetwork(self.statePh, actions)

