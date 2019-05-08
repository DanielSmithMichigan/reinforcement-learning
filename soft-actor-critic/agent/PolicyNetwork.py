import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

from . import constants
from . import util
from collections import deque

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
            theta,
            sigma
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
        self.buildNetwork()
        self.buildGraphs()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name=self.name + "_state")
            prevLayer = self.statePh
            for i in self.networkSize:
                prevLayer = tf.layers.dense(inputs=prevLayer, units=i, activation=tf.nn.relu)
                util.assertShape(prevLayer, [None, i])
            self.actionOutputMean = tf.layers.dense(inputs=prevLayer, units=self.numActions)
            util.assertShape(self.actionOutputMean, [None, self.numActions])
            self.actionOutputVariance = tf.abs(tf.layers.dense(inputs=prevLayer, units=self.numActions))
            util.assertShape(self.actionOutputVariance, [None, self.numActions])
            self.randomsPh = tf.placeholder(tf.float32, [None, self.numActions], name=self.name + "_randoms")
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros_like(self.actionOutputMean),
                scale_diag=tf.ones_like(self.actionOutputVariance)
            )
            distribution = (
                tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=tfp.bijectors.Affine(
                        shift=self.actionOutputMean,
                        scale_diag=tf.exp(self.actionOutputVariance)
                    )
                )
            )
            self.rawAction = self.actionOutputMean + (self.randomsPh * self.actionOutputVariance)
            self.actionsChosen = tf.nn.tanh(self.rawAction)
            util.assertShape(self.actionsChosen, [None, self.numActions])
            self.entropy = -(distribution.log_prob(self.rawAction))
            util.assertShape(self.entropy, [None])
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildGraphs(self):
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
        self.costOverTimeGraph.legend(loc=2)

        self.overview.canvas.draw()
    def setQNetwork(self, qNetwork):
        self.qNetwork = qNetwork
    def buildTrainingOperation(self):
        self.entropyCost = tf.reduce_mean(self.entropyCoefficient * self.entropy)
        util.assertShape(self.entropyCost, [])
        self.qCost = tf.reduce_mean(self.qNetwork.qValue)
        util.assertShape(self.qCost, [])
        self.cost = tf.reduce_mean(self.entropyCost + self.qCost)
        util.assertShape(self.cost, [])
        self.costActionGradient = tf.gradients(self.qCost, self.qNetwork.actionsPh)[0]
        self.qGradient = tf.gradients(self.actionsChosen, self.networkParams, -self.costActionGradient)
        self.entropyGradient = tf.gradients(self.entropyCost, self.networkParams)
        self.totalGradient = self.qGradient + self.entropyGradient
        (
            self.clippedGradients,
            self.gradientNorm
        ) = tf.clip_by_global_norm(self.qGradient, self.maxGradientNorm)
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.totalGradient, self.networkParams))
    def trainAgainst(self, memories):
        randoms = np.random.normal(loc=0.0, scale=1.0, size=(self.batchSize, self.numActions))
        actionsChosen = self.sess.run(self.actionsChosen, feed_dict={
            self.statePh: util.getColumn(memories, constants.STATE),
            self.randomsPh: randoms
        })
        (
            _,
            gradientNorm,
            qCost,
            entropyCost,
            qValue,
            entropy,
            qGradient
        ) = self.sess.run([
            self.trainingOperation,
            self.gradientNorm,
            self.qCost,
            self.entropyCost,
            self.qNetwork.qValue,
            self.entropy,
            self.costActionGradient
        ], feed_dict={
            self.statePh: util.getColumn(memories, constants.STATE),
            self.qNetwork.statePh: util.getColumn(memories, constants.STATE),
            self.qNetwork.actionsPh: actionsChosen,
            self.randomsPh: randoms
        })
        self.qCostOverTime.append(abs(qCost))
        self.entropyCostOverTime.append(abs(entropyCost))
        self.recentEntropy.append(np.mean(entropy))
        self.regTermOverTime.append(gradientNorm)
        self.entropyOverTime.append(np.mean(entropy))
    def getAction(self, state):
        randoms = np.random.normal(loc=0.0, scale=1.0, size=(1, self.numActions))
        (
            rawAction,
            output,
            entropy,
            actionOutputMean,
            actionOutputVariance
        ) = self.sess.run([
            self.rawAction,
            self.actionsChosen,
            self.entropy,
            self.actionOutputMean,
            self.actionOutputVariance
        ], feed_dict={
            self.statePh: [state],
            self.randomsPh: randoms
        })
        # print("MEAN: ",actionOutputMean[0][0]," VARIANCE: ",actionOutputVariance[0][0]," RAW: ",rawAction[0][0]," ENTROPY: ",entropy)
        self.actionsChosenOverTime.append(rawAction[0])
        self.meanOverTime.append(actionOutputMean[0])
        self.varianceOverTime.append(actionOutputVariance[0])
        self.actionOutputOverTime.append(output[0])
        return output[0]

