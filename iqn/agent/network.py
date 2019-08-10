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
            targetNetwork=None
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
        self.targetNetwork = targetNetwork
        self.losses = []
        self.build()
        if self.showGraph:
            self.buildGraphs()
    def build(self):
        with tf.variable_scope(self.name):
            self.weightsInitializer = slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
            self.environmentInput = prevLayer = tf.placeholder(tf.float32, [None, self.numObservations], "EnvironmentInput")
            # Shape: batchSize x numObservations
            for i in range(len(self.preNetworkSize)):
                prevLayer = tf.layers.dense(inputs=prevLayer, units=self.preNetworkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=self.weightsInitializer, name="pre_hidden_"+str(i))
            # Shape: batchSize x N
            self.buildEmbedding()
            # Shape: batchSize x numQuantiles x embeddingDimension
            self.embeddedQuantiles = tf.layers.dense(inputs=self.comparableQuantiles, units=self.preNetworkSize[-1], activation=tf.nn.leaky_relu, kernel_initializer=self.weightsInitializer)
            # Shape: batchSize x numQuantiles x N
            prevLayer = tf.reshape(prevLayer, [-1, 1, self.preNetworkSize[-1]])
            # Shape: batchSize x 1            x N
            self.firstJoinedLayer = prevLayer = self.embeddedQuantiles * prevLayer
            # Shape: batchSize x numQuantiles x N
            for i in range(len(self.postNetworkSize)):
                prevLayer = tf.layers.dense(inputs=prevLayer, units=self.postNetworkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=self.weightsInitializer, name="post_hidden_"+str(i))
            # Shape: batchSize x numQuantiles x N
            self.quantileValues = tf.layers.dense(inputs=prevLayer, units=self.numAvailableActions, kernel_initializer=self.weightsInitializer)
            # Shape: batchSize x numQuantiles x numActions
            self.quantileValues = tf.transpose(self.quantileValues, [0, 2, 1])
            # Shape: batchSize x numActions x numQuantiles
        self.networkParams = tf.trainable_variables(scope=self.name)
        self.buildMaxQHead()
        self.buildIndexedQuantiles()
        if self.targetNetwork:
            self.buildTrainingOperation()
    def buildEmbedding(self):
        self.quantileThresholds = tf.placeholder(tf.float32, [None, self.numQuantiles], "InputRandoms")
        # Shape: batchSize x numQuantiles
        self.reshapedQuantileThresholdsBox = tf.reshape(self.quantileThresholds, [-1, self.numQuantiles, 1])
        # Shape: batchSize x numQuantiles x 1
        self.quantilesEmbeddingBox = tf.tile(self.reshapedQuantileThresholdsBox, [1, 1, self.embeddingDimension])
        # Shape: batchSize x numQuantiles x embeddingDimension
        self.timestep = tf.reshape(tf.pow(2.0, tf.cast(tf.range(self.embeddingDimension), tf.float32)), [1, 1, self.embeddingDimension])
        # Shape: 1         x 1            x embeddingDimension
        self.tiledQuantiles = self.quantilesEmbeddingBox * self.timestep
        # Shape: batchSize x numQuantiles x embeddingDimension
        self.tiledQuantiles = self.tiledQuantiles - tf.math.floor(self.tiledQuantiles)
        # Shape: batchSize x numQuantiles x embeddingDimension
        self.comparableQuantiles = self.tiledQuantiles
        # Shape: batchSize x numQuantiles x embeddingDimension
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
    def buildMaxQHead(self):
        self.qValues = tf.reduce_mean(self.quantileValues, axis=2)
        # Shape: batchSize x numActions
        self.maxQ = tf.reduce_max(self.qValues, axis=1)
        self.chosenAction = tf.argmax(self.qValues, axis=1)
        batchIndices = tf.cast(tf.range(tf.shape(self.qValues)[0]), tf.int64)
        self.chosenActionQValue = tf.gather_nd(self.qValues, tf.stack([batchIndices, tf.cast(self.chosenAction, tf.int64)], axis=1))
    def buildIndexedQuantiles(self):
        self.actionInput = tf.placeholder(tf.int32, [None,], name="ActionInput")
        # Shape: batchSize
        self.indexedQuantiles = tf.gather_nd(self.quantileValues, tf.cast(tf.stack([tf.range(tf.shape(self.quantileValues)[0]), self.actionInput], axis=1), tf.int64))
        # Shape: batchSize x numQuantiles
    def buildTrainingOperation(self):
        self.gammas = tf.placeholder(tf.float32, [None,], name="Gammas")
        # Shape: batchSize
        self.observedRewards = tf.placeholder(tf.float32, [None,], name="ObservedRewards")
        # Shape: batchSize
        self.memoryPriority = tf.placeholder(tf.float32, [None,], name="MemoryPriority")
        # Shape: batchSize
        comparableRewards = tf.reshape(self.observedRewards, [-1, 1, 1])
        # Shape: batchSize x 1 x 1
        comparableGammas = tf.reshape(self.gammas, [-1, 1, 1])
        # Shape: batchSize x 1 x 1
        self.reshapedIndexedTargetQuantiles = tf.reshape(tf.stop_gradient(self.targetNetwork.indexedQuantiles), [-1, 1, self.numQuantiles])
        # Shape: batchSize x 1 x numTargetQuantiles
        self.tiledTargetValues = tf.tile(self.reshapedIndexedTargetQuantiles, [1, self.numQuantiles, 1])
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.targetValues = self.tiledTargetValues * comparableGammas + comparableRewards
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.reshapedIndexedQuantiles = tf.reshape(self.indexedQuantiles, [-1, self.numQuantiles, 1])
        # Shape: batchSize x numQuantiles x 1
        self.predictedValues = tf.tile(self.reshapedIndexedQuantiles, [1, 1, self.numQuantiles])
        # Shape: batchSize x numQuantiles x 1
        self.quantileDistance = self.targetValues - self.predictedValues
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.absQuantileDistance = tf.abs(self.quantileDistance)
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.minorQuantileError = tf.stop_gradient(tf.to_float(self.absQuantileDistance <= self.kappa)) * 0.5 * self.absQuantileDistance ** 2
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.majorQuantileError = tf.stop_gradient(tf.to_float(self.absQuantileDistance > self.kappa)) * self.kappa * (self.absQuantileDistance - 0.5 * self.kappa)
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.totalQuantileError = self.minorQuantileError + self.majorQuantileError
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.belowQuantile = tf.stop_gradient(tf.to_float(self.targetValues < self.predictedValues))
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.sizedQuantiles = tf.abs(self.reshapedQuantileThresholdsBox - self.belowQuantile)
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.quantileRegressionLoss = self.sizedQuantiles * self.totalQuantileError / self.kappa
        # Shape: batchSize x numQuantiles x numTargetQuantiles
        self.sumQuantileLoss = tf.reduce_mean(self.quantileRegressionLoss, axis=2)
        self.batchwiseLoss = tf.reduce_mean(self.sumQuantileLoss, axis=1)
        # Shape: batchSize
        self.proportionedLoss = (self.batchwiseLoss / self.memoryPriority)
        # Shape: batchSize
        self.finalLoss = tf.reduce_mean(self.proportionedLoss)
        # Shape: 1
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.finalLoss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.gradients, variables))
    def buildSoftCopyOperation(self, networkParams, tau):
        return [tf.assign(t, (1 - tau) * t + tau * e) for t, e in zip(self.networkParams, networkParams)]
    def getAction(self, state):
        (
            actionChosen,
            quantileValues,
            quantileThresholds,
            qValues
        ) = self.sess.run([
            self.chosenAction,
            self.quantileValues,
            self.quantileThresholds,
            self.qValues
        ], {
            self.environmentInput: [state],
            self.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(1, self.numQuantiles))
        })
        if self.showGraph:
            self.actionReasoningPlot.cla()
            self.actionReasoningPlot.set_title("Action Reasoning")
            for i in range(self.numAvailableActions):
                self.actionReasoningPlot.scatter(quantileThresholds, quantileValues[0][i], label=constants.ACTION_NAMES[i])
            self.actionReasoningPlot.legend(loc=2)
            self.actionReasoning.canvas.draw()
            self.actionChoiceBar.cla()
            self.actionChoiceBar.bar(constants.ACTION_NAMES, qValues[0])
            self.actionChoiceBar.set_ylabel("Q Value")
        # plt.pause(1)
        return actionChosen[0]
    def trainAgainst(self, memoryUnits):
        actions = util.getColumn(memoryUnits, constants.ACTION)
        nextActions = self.sess.run(self.chosenAction, feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.NEXT_STATE),
            self.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(self.batchSize, self.numQuantiles))
        })
        (targets,
            predictions,
            batchwiseLoss,
            finalLoss,
            _,
            learnedQuantileThresholdsOut,
            learnedQuantileValues,
            targetQuantileThresholdsOut,
            targetQuantileValues
        ) = self.sess.run([
            self.targetValues,
            self.predictedValues,
            self.batchwiseLoss,
            self.finalLoss,
            self.trainingOperation,
            self.quantileThresholds,
            self.quantileValues,
            self.targetNetwork.quantileThresholds,
            self.targetNetwork.quantileValues
        ], feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.STATE),
            self.memoryPriority: util.getColumn(memoryUnits, constants.PRIORITY),
            self.actionInput: actions,
            self.observedRewards: util.getColumn(memoryUnits, constants.REWARD),
            self.gammas: util.getColumn(memoryUnits, constants.GAMMA),
            self.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(self.batchSize, self.numQuantiles)),
            self.targetNetwork.environmentInput: util.getColumn(memoryUnits, constants.NEXT_STATE),
            self.targetNetwork.actionInput: nextActions,
            self.targetNetwork.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(self.batchSize, self.numQuantiles))
        })
        selectedBatch = np.random.randint(self.batchSize)
        self.previousLearnedQuantileThresholds = learnedQuantileThresholdsOut[selectedBatch]
        self.previousLearnedQuantileValues = learnedQuantileValues[selectedBatch][actions[selectedBatch]]
        self.previousTargetQuantileThresholds = targetQuantileThresholdsOut[selectedBatch]
        self.previousTargetQuantileValues = targetQuantileValues[selectedBatch][nextActions[selectedBatch]]
        self.losses.append(finalLoss)
        for i in range(len(memoryUnits)):
            memoryUnits[i][constants.LOSS] = batchwiseLoss[i]
        return targets, predictions, actions