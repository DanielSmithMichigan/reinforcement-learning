import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

from . import constants
from . import util
from collections import deque

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


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
        self.regTermOverTime = []
        self.entropyOverTime = []
        self.actionsChosenOverTime = []
        self.meanOverTime = []
        self.varianceOverTime = []
        self.entropyCostOverTime = []
        self.recentEntropy = deque([], 100)
        self.qCostOverTime = []
        self.explorationOverTime = []
        self.actionOutputOverTime = []
        self.actionNoise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(numActions), theta=theta, sigma=sigma)
        self.buildNetwork()
        self.buildGraphs()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name=self.name + "_state")
            prevLayer = self.statePh
            for i in self.networkSize:
                prevLayer = tf.layers.dense(inputs=prevLayer, units=i, activation=tf.nn.leaky_relu)
                util.assertShape(prevLayer, [None, i])
            self.actionOutputMean = tf.layers.dense(inputs=prevLayer, units=self.numActions)
            util.assertShape(self.actionOutputMean, [None, self.numActions])
            self.actionOutputVariance = tf.abs(tf.layers.dense(inputs=prevLayer, units=self.numActions))
            util.assertShape(self.actionOutputVariance, [None, self.numActions])
            self.randomsPh = tf.placeholder(tf.float32, [None, self.numActions], name=self.name + "_randoms")
            self.explorationPh = tf.placeholder(tf.float32, [None, self.numActions], name=self.name + "_exploration")
            self.rawAction = self.actionOutputMean + (self.randomsPh * self.actionOutputVariance)
            self.actionsChosen = tf.nn.tanh(self.rawAction + self.explorationPh)
            util.assertShape(self.actionsChosen, [None, self.numActions])
            self.normalDistribution = tf.distributions.Normal(self.actionOutputMean, self.actionOutputVariance)
            self.entropy = -self.normalDistribution.log_prob(self.rawAction)
            util.assertShape(self.entropy, [None, self.numActions])
            self.entropy = tf.reduce_sum(self.entropy)
            util.assertShape(self.entropy, [])
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
        self.actionChoicesGraph.plot(util.getColumn(self.actionsChosenOverTime, 0), label="choice")
        self.actionChoicesGraph.plot(util.getColumn(self.meanOverTime, 0), label="mean")
        self.actionChoicesGraph.plot(util.getColumn(self.varianceOverTime, 0), label="variance")
        self.actionChoicesGraph.plot(util.getColumn(self.explorationOverTime, 0), label="exploration")
        self.actionChoicesGraph.plot(util.getColumn(self.actionOutputOverTime, 0), label="actionOutput")
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
            self.randomsPh: randoms,
            self.explorationPh: np.zeros((self.batchSize, self.numActions))
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
            self.randomsPh: randoms,
            self.explorationPh: np.zeros((self.batchSize, self.numActions))
        })
        self.qCostOverTime.append(abs(qCost))
        self.entropyCostOverTime.append(abs(entropyCost))
        self.recentEntropy.append(entropy)
        self.regTermOverTime.append(gradientNorm)
        self.entropyOverTime.append(entropy)
    def getAction(self, state):
        randoms = np.random.normal(loc=0.0, scale=1.0, size=(1, self.numActions))
        actionNoise = np.reshape(self.actionNoise(), [-1, 1])
        actionNoise = np.zeros((self.batchSize, self.numActions))
        (
            exploration,
            rawAction,
            output,
            entropy,
            actionOutputMean,
            actionOutputVariance
        ) = self.sess.run([
            self.explorationPh,
            self.rawAction,
            self.actionsChosen,
            self.entropy,
            self.actionOutputMean,
            self.actionOutputVariance
        ], feed_dict={
            self.statePh: [state],
            self.randomsPh: randoms,
            self.explorationPh: actionNoise
        })
        # print("MEAN: ",actionOutputMean[0][0]," VARIANCE: ",actionOutputVariance[0][0]," RAW: ",rawAction[0][0]," ENTROPY: ",entropy)
        self.actionsChosenOverTime.append(rawAction[0])
        self.meanOverTime.append(actionOutputMean[0])
        self.varianceOverTime.append(actionOutputVariance[0])
        self.explorationOverTime.append(exploration[0])
        self.actionOutputOverTime.append(output[0])
        return output[0]

