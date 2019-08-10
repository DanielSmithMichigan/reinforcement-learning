import numpy as np
import tensorflow as tf
from c51.distribution import buildTargets
from .noise import noisy_dense_layer
from . import constants
from . import util
import math
slim = tf.contrib.slim
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

weightsInitializer = slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)


class Network:
    def __init__(self,
            name,
            sess,
            showGraph,
            numObservations,
            preNetworkSize,
            postNetworkSize,
            numQuantiles,
            numAvailableActions,
            embeddingDimension,
            learningRate,
            maxGradientNorm,
            batchSize,
            kappa,
            statePh,
            nextStatePh,
            actionsPh,
            rewardsPh,
            gammasPh,
            quantileThresholdsPh,
            nextQuantileThresholdsPh
        ):
        self.name = name
        self.sess = sess
        self.showGraph = showGraph
        self.numAvailableActions = numAvailableActions
        self.numObservations = numObservations
        self.preNetworkSize = preNetworkSize
        self.postNetworkSize = postNetworkSize
        self.numQuantiles = numQuantiles
        self.embeddingDimension = embeddingDimension
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.kappa = kappa
        self.maxGradientNorm = maxGradientNorm
        self.statePh=statePh
        self.nextStatePh=nextStatePh
        self.actionsPh=actionsPh
        self.rewardsPh=rewardsPh
        self.gammasPh=gammasPh
        self.quantileThresholdsPh=quantileThresholdsPh
        self.nextQuantileThresholdsPh=nextQuantileThresholdsPh
        self.losses = []
        if self.showGraph:
            self.buildGraphs()
    def buildNetwork(
        self,
        statePh,
        quantileThresholds,
        actionsPh
    ):
        with tf.variable_scope(self.name):
            prevLayer = statePh
            for i in range(len(self.preNetworkSize)):
                prevLayer = tf.layers.dense(
                    inputs=prevLayer,
                    units=self.preNetworkSize[i],
                    activation=tf.nn.leaky_relu,
                    name="pre_hidden_"+str(i),
                    reuse=tf.AUTO_REUSE,
                    kernel_initializer=weightsInitializer
                )
            reshapedQuantileThresholdsBox = tf.reshape(quantileThresholds, [-1, self.numQuantiles, 1])
            quantilesEmbeddingBox = tf.tile(reshapedQuantileThresholdsBox, [1, 1, self.embeddingDimension])
            timestep = tf.pow(2.0, tf.cast(tf.range(self.embeddingDimension), tf.float32))
            timestep = tf.reshape(timestep, [1, 1, self.embeddingDimension])
            tiledQuantiles = quantilesEmbeddingBox * timestep
            tiledQuantiles = tiledQuantiles - tf.math.floor(tiledQuantiles)
            embeddedQuantiles = tf.layers.dense(
                inputs=tiledQuantiles,
                units=self.preNetworkSize[-1],
                activation=tf.nn.leaky_relu,
                name="embedding_hidden",
                reuse=tf.AUTO_REUSE,
                kernel_initializer=weightsInitializer
            )
            prevLayer = tf.reshape(prevLayer, [-1, 1, self.preNetworkSize[-1]])
            prevLayer = embeddedQuantiles * prevLayer
            for i in range(len(self.postNetworkSize)):
                prevLayer = tf.layers.dense(
                    inputs=prevLayer,
                    units=self.postNetworkSize[i],
                    activation=tf.nn.leaky_relu,
                    name="post_hidden_"+str(i),
                    reuse=tf.AUTO_REUSE,
                    kernel_initializer=weightsInitializer
                )
            quantileValues = tf.layers.dense(
                inputs=prevLayer,
                units=self.numAvailableActions,
                name="quantile_values",
                reuse=tf.AUTO_REUSE,
                kernel_initializer=weightsInitializer
            )
            quantileValues = tf.transpose(quantileValues, [0, 2, 1])
            qValues = tf.reduce_mean(quantileValues, axis=2)
            maxQ = tf.reduce_max(qValues, axis=1)
            chosenAction = tf.argmax(qValues, axis=1)
            chosenAction = tf.cast(chosenAction, tf.int32)
            batchSize = tf.shape(qValues)[0]
            batchIndices = tf.cast(tf.range(batchSize), tf.int32)
            chosenActionQValue = tf.gather_nd(qValues, tf.stack([batchIndices, chosenAction], axis=1))
            switch = actionsPh if (actionsPh != None) else chosenAction
            idx = tf.stack([batchIndices, switch], axis=1)        
            indexedQuantiles = tf.gather_nd(quantileValues, idx)
        return (
            quantileValues,
            qValues,
            maxQ,
            chosenAction,
            chosenActionQValue,
            indexedQuantiles
        )
    def buildGraphs(self):
        self.overview = plt.figure()
        self.previousEvaluationLearned = self.overview.add_subplot(2, 1, 1)
        self.previousLearnedQuantileThresholds = np.zeros((self.numQuantiles,))
        self.previousLearnedQuantileValues = np.zeros((self.numQuantiles,))

        self.previousEvaluationTarget = self.overview.add_subplot(2, 1, 2)
        self.previousTargetQuantileThresholds = np.zeros((self.numQuantiles,))
        self.previousTargetQuantileValues = np.zeros((self.numQuantiles,))

        self.actionReasoning = plt.figure()
        self.actionReasoningPlot = self.actionReasoning.add_subplot(2, 1, 1)
        self.actionChoiceBar = self.actionReasoning.add_subplot(2, 1, 2)
    def updateGraphs(self):
        self.previousEvaluationLearned.cla()
        self.previousEvaluationLearned.set_title("Learned")
        self.previousEvaluationLearned.scatter(self.previousLearnedQuantileThresholds, self.previousLearnedQuantileValues)
        self.previousEvaluationTarget.cla()
        self.previousEvaluationTarget.set_title("Target")
        self.previousEvaluationTarget.scatter(self.previousTargetQuantileThresholds, self.previousTargetQuantileValues)
        self.overview.canvas.draw()
    def setTargetNetwork(self, targetNetwork):
        self.targetNetwork = targetNetwork
    def buildTrainingOperation(self):
        (
            _,
            _,
            _,
            _,
            _,
            indexedQuantilesNextState
        ) = self.targetNetwork.buildNetwork(
            self.nextStatePh,
            self.nextQuantileThresholdsPh,
            None
        )
        (
            _,
            _,
            _,
            _,
            _,
            indexedQuantilesCurrentState
        ) = self.buildNetwork(
            self.statePh,
            self.quantileThresholdsPh,
            self.actionsPh
        )

        # Get targets
        reshapedIndexedTargetQuantiles = tf.reshape(tf.stop_gradient(indexedQuantilesNextState), [-1, 1, self.numQuantiles])
        tiledTargetValues = tf.tile(reshapedIndexedTargetQuantiles, [1, self.numQuantiles, 1])
        comparableRewards = tf.reshape(self.rewardsPh, [-1, 1, 1])
        comparableGammas = tf.reshape(self.gammasPh, [-1, 1, 1])
        targetValues = tiledTargetValues * comparableGammas + comparableRewards
        # Shape: batchSize x numQuantiles (tiled) x numQuantiles

        # Get predictions
        reshapedIndexedQuantiles = tf.reshape(indexedQuantilesCurrentState, [-1, self.numQuantiles, 1])
        predictedValues = tf.tile(reshapedIndexedQuantiles, [1, 1, self.numQuantiles])
        # Shape: batchSize x numQuantiles x numQuantiles (tiled)

        quantileDistance = targetValues - predictedValues
        absQuantileDistance = tf.abs(quantileDistance)
        minorQuantileError = tf.stop_gradient(tf.to_float(absQuantileDistance <= self.kappa)) * 0.5 * absQuantileDistance ** 2
        majorQuantileError = tf.stop_gradient(tf.to_float(absQuantileDistance > self.kappa)) * self.kappa * (absQuantileDistance - 0.5 * self.kappa)
        totalQuantileError = minorQuantileError + majorQuantileError
        belowQuantile = tf.stop_gradient(tf.to_float(targetValues < predictedValues))
        reshapedQuantileThresholds = tf.reshape(self.quantileThresholdsPh, [-1, self.numQuantiles, 1])
        reshapedQuantileThresholds = tf.tile(reshapedQuantileThresholds, [1, 1, self.numQuantiles])
        # Shape: batchSize x numQuantiles x numQuantiles (tiled)
        sizedQuantiles = tf.abs(reshapedQuantileThresholds - belowQuantile)
        quantileRegressionLoss = sizedQuantiles * totalQuantileError / self.kappa
        sumQuantileLoss = tf.reduce_mean(quantileRegressionLoss, axis=2)
        batchwiseLoss = tf.reduce_mean(sumQuantileLoss, axis=1)
        finalLoss = tf.reduce_mean(batchwiseLoss)
        optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(
            *optimizer.compute_gradients(
                finalLoss,
                var_list=tf.trainable_variables(scope=self.name)
            )
        )
        gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        trainingOperation = optimizer.apply_gradients(
            zip(gradients, variables)
        )
        return (
            finalLoss,
            trainingOperation
        )
    def buildSoftCopyOperation(self, otherNetwork, tau):
        return [tf.assign(t, (1 - tau) * t + tau * e) for t, e in zip(
            tf.trainable_variables(
                scope=self.name
            ),
            tf.trainable_variables(
                scope=otherNetwork.name
            )
        )]