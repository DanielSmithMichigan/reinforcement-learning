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
            preNetworkSize,
            postNetworkSize,
            numQuantiles,
            embeddingDimension,
            gamma,
            kappa,
            learningRate,
            showGraphs,
            statePh,
            nextStatePh,
            actionsPh,
            rewardsPh,
            terminalsPh,
            quantileThresholdsPh,
            nextQuantileThresholdsPh,
            memoryPriorityPh,
            maxGradientNorm
        ):
        self.sess = sess
        self.graph = graph
        self.statePh = statePh
        self.nextStatePh = nextStatePh
        self.actionsPh = actionsPh
        self.rewardsPh = rewardsPh
        self.quantileThresholdsPh = quantileThresholdsPh
        self.nextQuantileThresholdsPh = nextQuantileThresholdsPh
        self.terminalsPh = terminalsPh
        self.memoryPriorityPh = memoryPriorityPh
        self.maxGradientNorm = maxGradientNorm
        self.name = name
        self.numStateVariables = numStateVariables
        self.numActions = numActions
        self.preNetworkSize = preNetworkSize
        self.postNetworkSize = postNetworkSize
        self.numQuantiles = numQuantiles
        self.embeddingDimension = embeddingDimension
        self.gamma = gamma
        self.kappa = kappa
        self.learningRate = learningRate
    def buildNetwork(self, state, actions, quantileThresholds):
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
                for i in range(len(self.preNetworkSize)):
                    prevLayer = tf.layers.dense(
                        inputs=prevLayer,
                        units=self.preNetworkSize[i],
                        activation=tf.nn.leaky_relu,
                        name="dense_pre_" + str(i),
                        reuse=tf.AUTO_REUSE
                    )
                # batchSize x N
                prevLayer = tf.reshape(prevLayer, [-1, 1, self.preNetworkSize[-1]])
                # batchSize x 1 x N
                (
                    quantileEmbedding,
                    _,
                    _,
                    _,
                    _,
                ) = self.buildQuantileEmbeddingNetwork(quantileThresholds)
                prevLayer = quantileEmbedding * prevLayer
                # batchSize x numQuantiles x N
                for i in range(len(self.postNetworkSize)):
                    prevLayer = tf.layers.dense(
                        inputs=prevLayer,
                        units=self.postNetworkSize[i],
                        activation=tf.nn.leaky_relu,
                        name="dense_post_" + str(i),
                        reuse=tf.AUTO_REUSE
                    )
                # batchSize x numQuantiles x N
                quantileValues = tf.layers.dense(
                    inputs=prevLayer,
                    units=1,
                    name="qValue",
                    reuse=tf.AUTO_REUSE
                )
                # batchSize x numQuantiles x 1
                quantileValues = tf.reshape(quantileValues, [-1, self.numQuantiles])
                # batchSize x numQuantiles
                qValue = tf.reduce_mean(quantileValues, axis=1)
                # batchSize
        return quantileValues, qValue
    def buildQuantileEmbeddingNetwork(self, quantileThresholds):
        quantileThresholds = tf.reshape(quantileThresholds, [-1, self.numQuantiles, 1])
        # batchSize x numQuantiles x 1
        tiledQuantiles = tf.tile(quantileThresholds, [1, 1, self.embeddingDimension])
        # batchSize x numQuantiles x embeddingDimension
        timestep = tf.pow(2.0, tf.cast(tf.range(self.embeddingDimension), tf.float32))
        timestep = tf.reshape(timestep, [1, 1, self.embeddingDimension])
        # 1 x 1 x embeddingDimension
        portionedQuantiles = tiledQuantiles * timestep
        # batchSize x numQuantiles x embeddingDimension
        sawtooth = portionedQuantiles - tf.math.floor(portionedQuantiles)
        # batchSize x numQuantiles x embeddingDimension
        quantileEmbedding = tf.layers.dense(
            inputs=sawtooth,
            units=self.preNetworkSize[-1],
            activation=tf.nn.leaky_relu,
            name="dense_embedding",
            reuse=tf.AUTO_REUSE
        )
        # batchSize x numQuantiles x N
        return (
            quantileEmbedding,
            tiledQuantiles,
            timestep,
            portionedQuantiles,
            sawtooth
        )
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
    def setTargetNetworks(self, target1):
        self.target1 = target1
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
            ) = self.target1.buildNetwork(self.nextStatePh, nextActionsChosen, self.nextQuantileThresholdsPh)
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
            ) = self.buildNetwork(self.statePh, self.actionsPh, self.quantileThresholdsPh)
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
            quantileThresholds = tf.reshape(self.quantileThresholdsPh, [-1, self.numQuantiles, 1])
            # batchSize x numQuantiles x 1
            quantilePunishment = tf.abs(quantileThresholds - tf.stop_gradient(tf.to_float(targets < predictions)))
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
            self.assessmentOperation = self.buildNetwork(self.statePh, actions, self.quantileThresholdsPh)

