import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

import util
import constants

class ValueNetwork:
    def __init__(self,
            sess,
            name,
            numStateVariables,
            networkSize,
            entropyCoefficient,
            learningRate,
            maxGradientNorm,
            batchSize,
            numActions
        ):
        self.sess = sess
        self.name = name
        self.numStateVariables = numStateVariables
        self.networkSize = networkSize
        self.entropyCoefficient = entropyCoefficient
        self.learningRate = learningRate
        self.maxGradientNorm = maxGradientNorm
        self.batchSize = batchSize
        self.numActions = numActions
        self.lossOverTime = []
        self.targetValueOverTime = []
        self.predictedValueOverTime = []
        self.entropyValueOverTime = []
        self.buildNetwork()
        self.buildGraphs()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables])
            util.assertShape(self.statePh, [None, self.numStateVariables])
            prevLayer = self.statePh
            for i in self.networkSize:
                prevLayer = tf.layers.dense(inputs=prevLayer, units=i, activation=tf.nn.leaky_relu)
                util.assertShape(prevLayer, [None, i])
            self.value = tf.layers.dense(inputs=prevLayer, units=1)
            util.assertShape(self.value, [None, 1])
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildGraphs(self):
        self.overview = plt.figure()
        self.overview.suptitle(self.name)
        self.lossGraph = self.overview.add_subplot(2, 1, 1)
        self.predictedVsActualGraph = self.overview.add_subplot(2, 1, 2)
    def updateGraphs(self):
        self.lossGraph.cla()
        self.lossGraph.plot(self.lossOverTime)
        self.lossGraph.set_title("Loss")

        self.predictedVsActualGraph.cla()
        self.predictedVsActualGraph.set_title("Predicted Vs Actual")
        self.predictedVsActualGraph.plot(self.targetValueOverTime, label="Target Value")
        self.predictedVsActualGraph.plot(self.predictedValueOverTime, label="Predicted Value")
        self.predictedVsActualGraph.plot(self.entropyValueOverTime, label="Entropy")
        self.predictedVsActualGraph.legend(loc=2)

        self.overview.canvas.draw()
    def buildSoftCopyOperation(self, networkParams, tau):
        return [tf.assign(t, (1 - tau) * t + tau * e) for t, e in zip(self.networkParams, networkParams)]
    def setNetworks(self, policyNetwork, qNetwork1, qNetwork2):
        self.policyNetwork = policyNetwork
        self.qNetwork1 = qNetwork1
        self.qNetwork2 = qNetwork2
    def buildTrainingOperation(self):
        self.minQValue = tf.minimum(self.qNetwork1.qValue, self.qNetwork2.qValue)
        util.assertShape(self.minQValue, [None, 1])
        self.entropyValue = self.entropyCoefficient * self.policyNetwork.entropy
        util.assertShape(self.entropyValue, [])
        self.targetValue = self.minQValue + self.entropyValue
        util.assertShape(self.targetValue, [None, 1])
        self.targetValuePh = tf.placeholder(tf.float32, [None, 1], name="TargetValuePlaceholder")
        self.loss = tf.reduce_mean(tf.pow(self.value - self.targetValuePh, 2))
        util.assertShape(self.loss, [])
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        (
            self.gradients,
            self.gradientNorm
        ) = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.gradients, variables))
    def getTargets(self, memories):
        actionsChosen = self.sess.run(self.policyNetwork.actionsChosen, feed_dict={
            self.policyNetwork.statePh: util.getColumn(memories, constants.STATE),
            self.policyNetwork.randomsPh: np.random.normal(loc=0.0, scale=1.0, size=(self.batchSize, self.numActions)),
            self.policyNetwork.explorationPh: np.zeros((self.batchSize, self.numActions))
        })
        targetValues, entropy = self.sess.run([self.targetValue, self.entropyValue], feed_dict={
            self.qNetwork1.statePh: util.getColumn(memories, constants.STATE),
            self.qNetwork1.actionsPh: actionsChosen,
            self.qNetwork2.statePh: util.getColumn(memories, constants.STATE),
            self.qNetwork2.actionsPh: actionsChosen,
            self.policyNetwork.statePh: util.getColumn(memories, constants.STATE),
            self.policyNetwork.explorationPh: np.zeros((self.batchSize, self.numActions)),
            self.policyNetwork.randomsPh: np.random.normal(loc=0.0, scale=1.0, size=(self.batchSize, self.numActions))
        })
        for i in targetValues:
            self.targetValueOverTime.append(i[0])
        self.entropyValueOverTime.append(entropy)
        return targetValues
    def trainAgainst(self, memories, targets):
        loss, _, predictedValues = self.sess.run([self.loss, self.trainingOperation, self.value], feed_dict={
            self.targetValuePh: targets,
            self.statePh: util.getColumn(memories, constants.STATE)
        })

        for i in predictedValues:
            self.predictedValueOverTime.append(i[0])
        self.lossOverTime.append(loss)

