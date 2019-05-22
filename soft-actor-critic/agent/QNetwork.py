import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import util
from . import constants

class QNetwork:
    def __init__(self,
            sess,
            name,
            numStateVariables,
            numActions,
            networkSize,
            gamma,
            learningRate,
            maxGradientNorm,
            showGraphs
        ):
        self.sess = sess
        self.name = name
        self.numStateVariables = numStateVariables
        self.numActions = numActions
        self.networkSize = networkSize
        self.gamma = gamma
        self.learningRate = learningRate
        self.maxGradientNorm = maxGradientNorm
        self.storedTargets = None
        self.lossOverTime = []
        self.assessmentsOverTime = []
        if showGraphs:
            self.buildGraphs()
    def buildNetwork(self):
        with tf.variable_scope(self.name):
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables])
            self.stateEmbedding = tf.layers.dense(inputs=self.statePh, units=self.networkSize[0], activation=tf.nn.relu, name=self.name+"_state_embedding")
            self.actionsPh = self.policyNetwork.actionsChosen
            self.actionEmbedding = tf.layers.dense(inputs=self.actionsPh, units=self.networkSize[0], activation=tf.nn.relu, name=self.name+"_actions_embedding")
            prevLayer = tf.concat([self.stateEmbedding, self.actionEmbedding], axis=1)
            for i in range(1, len(self.networkSize)):
                prevLayer = tf.layers.dense(inputs=prevLayer, units=self.networkSize[i], activation=tf.nn.relu, name=self.name + "_dense_" + str(i))
                util.assertShape(prevLayer, [None, self.networkSize[i]])
            self.qValue = tf.layers.dense(inputs=prevLayer, units=1)
            util.assertShape(self.qValue, [None, 1])
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildGraphs(self):
        plt.ion()
        self.overview = plt.figure()
        self.overview.suptitle(self.name)
        self.lossGraph = self.overview.add_subplot(2, 1, 1)
        self.assessmentsGraph = self.overview.add_subplot(2, 1, 2)
    def updateGraphs(self):
        self.lossGraph.cla()
        self.lossGraph.set_title("loss")
        self.lossGraph.plot(self.lossOverTime)

        self.assessmentsGraph.cla()
        self.assessmentsGraph.set_title("assessments")
        self.assessmentsGraph.plot(self.assessmentsOverTime)

        self.overview.canvas.draw()
    def storeAssessment(self, state, actions):
        assessment = self.sess.run(self.qValue, feed_dict={
            self.statePh: [state],
            self.actionsPh: [actions]
        })
        self.assessmentsOverTime.append(assessment[0])
    def setValueNetwork(self, valueNetwork):
        self.valueNetwork = valueNetwork
    def setPolicyNetwork(self, policyNetwork):
        self.policyNetwork = policyNetwork
    def buildTrainingOperation(self):
        self.rewardsPh = tf.placeholder(tf.float32, [None])
        # util.assertShape(self.rewardsPh, [])
        self.terminalsPh = tf.to_float(tf.placeholder(tf.bool, [None]))
        # util.assertShape(self.terminalsPh, [])
        reshapedValue = tf.reshape(self.valueNetwork.value, [-1])
        # util.assertShape(reshapedValue, )
        self.targetQ = self.rewardsPh + self.gamma * (1 - self.terminalsPh) * reshapedValue
        util.assertShape(self.targetQ, [None,])
        self.targetQPh = tf.placeholder(tf.float32, [None,], name="TargetQPh")
        self.loss = tf.reduce_mean(tf.pow(self.qValue - self.targetQPh, 2))
        util.assertShape(self.loss, [])
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(
            *self.optimizer.compute_gradients(
                self.loss,
                var_list=self.networkParams
            )
        )
        (
            self.gradients,
            self.gradientNorm
        ) = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.gradients, variables))
    def getTargets(self, memories):
        rewards = util.getColumn(memories, constants.REWARD)
        targets = self.sess.run(self.targetQ, feed_dict={
            self.rewardsPh: rewards,
            self.terminalsPh: util.getColumn(memories, constants.IS_TERMINAL),
            self.valueNetwork.statePh: util.getColumn(memories, constants.NEXT_STATE),
            self.policyNetwork.statePh: util.getColumn(memories, constants.STATE)
        })
        return targets
    def trainAgainst(self, memories, targets):
        loss, _ = self.sess.run([self.loss, self.trainingOperation], feed_dict={
            self.targetQPh: targets,
            self.statePh: util.getColumn(memories, constants.STATE),
            self.actionsPh: util.getColumn(memories, constants.ACTION),
            self.policyNetwork.statePh: util.getColumn(memories, constants.STATE)
        })
        self.lossOverTime.append(loss)

