import tensorflow as tf
import numpy as np
import gym
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import time

from .PolicyNetwork import PolicyNetwork
from .QNetwork import QNetwork
from .ValueNetwork import ValueNetwork
from . import constants

from .prioritized_experience_replay import PrioritizedExperienceReplay


class Agent:
    def __init__(self,
            name,
            policyNetworkSize,
            qNetworkSize,
            valueNetworkSize,
            entropyCoefficient,
            valueNetworkLearningRate,
            policyNetworkLearningRate,
            qNetworkLearningRate,
            tau,
            gamma,
            maxMemoryLength,
            priorityExponent,
            batchSize,
            maxGradientNorm,
            maxEpisodes,
            trainSteps,
            rewardScaling,
            stepsPerUpdate,
            render,
            showGraphs,
            minStepsBeforeTraining,
            actionScaling,
            actionShift,
            meanRegularizationConstant,
            varianceRegularizationConstant,
            testSteps,
            maxMinutes
        ):
        self.numStateVariables = 3
        self.numActions = 1
        self.env = gym.make('Pendulum-v0')
        self.sess = tf.Session()
        self.startTime = time.time()

        self.learnedValueNetwork = ValueNetwork(
            sess=self.sess,
            name="LearnedValueNetwork_"+name,
            numStateVariables=self.numStateVariables,
            networkSize=valueNetworkSize,
            entropyCoefficient=entropyCoefficient,
            learningRate=valueNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            numActions=self.numActions,
            showGraphs=showGraphs
        )
        self.targetValueNetwork = ValueNetwork(
            name="TargetValueNetwork_"+name,
            sess=self.sess,
            numStateVariables=self.numStateVariables,
            networkSize=valueNetworkSize,
            entropyCoefficient=entropyCoefficient,
            learningRate=valueNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            numActions=self.numActions,
            showGraphs=showGraphs
        )
        self.copyLearnedNetwork = self.targetValueNetwork.buildSoftCopyOperation(self.learnedValueNetwork.networkParams, 1)
        self.softCopyLearnedNetwork = self.targetValueNetwork.buildSoftCopyOperation(self.learnedValueNetwork.networkParams, tau)

        self.qNetwork1 = QNetwork(
            sess=self.sess,
            name="QNetwork_1_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            showGraphs=showGraphs
        )
        self.qNetwork2 = QNetwork(
            sess=self.sess,
            name="QNetwork_2_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            showGraphs=showGraphs
        )

        self.policyNetwork = PolicyNetwork(
            sess=self.sess,
            name="PolicyNetwork_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=policyNetworkSize,
            entropyCoefficient=entropyCoefficient,
            learningRate=policyNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            meanRegularizationConstant=meanRegularizationConstant,
            varianceRegularizationConstant=varianceRegularizationConstant,
            showGraphs=showGraphs
        )

        self.qNetwork1.setValueNetwork(self.targetValueNetwork)
        self.qNetwork1.setPolicyNetwork(self.policyNetwork)
        self.qNetwork1.buildNetwork()
        self.qNetwork1.buildTrainingOperation()

        self.qNetwork2.setValueNetwork(self.targetValueNetwork)
        self.qNetwork2.setPolicyNetwork(self.policyNetwork)
        self.qNetwork2.buildNetwork()
        self.qNetwork2.buildTrainingOperation()

        self.learnedValueNetwork.setNetworks(self.policyNetwork, self.qNetwork1, self.qNetwork2)
        self.learnedValueNetwork.buildTrainingOperation()

        self.targetValueNetwork.setNetworks(self.policyNetwork, self.qNetwork1, self.qNetwork2)
        self.targetValueNetwork.buildTrainingOperation()

        self.policyNetwork.setQNetwork(self.qNetwork1)
        self.policyNetwork.buildTrainingOperation()

        self.memoryBuffer = PrioritizedExperienceReplay(
            numMemories=maxMemoryLength,
            priorityExponent=priorityExponent,
            batchSize=batchSize
        )

        self.name = name
        self.maxEpisodes = maxEpisodes
        self.trainSteps = trainSteps
        self.rewardScaling = rewardScaling
        self.gamma = gamma
        self.stepsPerUpdate = stepsPerUpdate
        self.render = render
        self.showGraphs = showGraphs
        self.minStepsBeforeTraining = minStepsBeforeTraining
        self.actionScaling = actionScaling
        self.actionShift = actionShift
        self.testSteps = testSteps
        self.maxMinutes = maxMinutes
        if showGraphs:
            self.buildGraphs()
        self.getQTargetsOverTime = deque([], 400)
        self.getValueTargetsOverTime = deque([], 400)
        self.trainQOverTime = deque([], 400)
        self.trainValueOverTime = deque([], 400)
        self.trainPolicyOverTime = deque([], 400)
        self.rewardsOverTime = deque([], 400)
    def buildGraphs(self):
        plt.ion()
        self.overview = plt.figure()
        self.overview.suptitle(self.name)
        self.timersPlot = self.overview.add_subplot(2, 1, 1)
        self.rewardsPlot = self.overview.add_subplot(2, 1, 2)
    def outOfTime(self):
        return time.time() > self.startTime + (self.maxMinutes * 60)
    def goToNextState(self):
        (
            actionsChosen,
            logScaleActionVariance,
            logProb,
            entropy
        ) = self.policyNetwork.getAction(self.state)
        self.qNetwork1.storeAssessment(self.state, actionsChosen)
        self.qNetwork2.storeAssessment(self.state, actionsChosen)
        nextState, reward, done, info = self.env.step((actionsChosen + self.actionShift) * self.actionScaling)
        done = False
        if (self.render):
            self.env.render()
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = self.state
        memoryEntry[constants.ACTION] = actionsChosen
        # memoryEntry[constants.REWARD] = np.reshape(-abs(actionsChosen - .1), [])
        memoryEntry[constants.REWARD] = reward * self.rewardScaling
        memoryEntry[constants.NEXT_STATE] = nextState
        memoryEntry[constants.GAMMA] = self.gamma if not done else 0
        memoryEntry[constants.IS_TERMINAL] = done
        self.state = nextState
        self.memoryBuffer.add(memoryEntry)
        self.globalStep = self.globalStep + 1
        self.totalEpisodeReward = self.totalEpisodeReward + reward
        if self.globalStep % self.stepsPerUpdate == 0 and self.globalStep > self.minStepsBeforeTraining:
            self.train()
        return done
    def train(self):
        trainingMemories = self.memoryBuffer.getMemoryBatch()

        start = time.time()
        qOneTargets = self.qNetwork1.getTargets(trainingMemories)
        qTwoTargets = self.qNetwork2.getTargets(trainingMemories)
        self.getQTargetsOverTime.append(time.time() - start)

        start = time.time()
        valueTargets = self.targetValueNetwork.getTargets(trainingMemories)
        self.getValueTargetsOverTime.append(time.time() - start)

        start = time.time()
        self.qNetwork1.trainAgainst(trainingMemories, qOneTargets)
        self.qNetwork2.trainAgainst(trainingMemories, qTwoTargets)
        self.trainQOverTime.append(time.time() - start)

        start = time.time()
        self.learnedValueNetwork.trainAgainst(trainingMemories, valueTargets)
        self.sess.run(self.softCopyLearnedNetwork)
        self.trainValueOverTime.append(time.time() - start)

        start = time.time()
        self.policyNetwork.trainAgainst(trainingMemories)
        self.trainPolicyOverTime.append(time.time() - start)
    def updateGraphs(self):
        self.qNetwork1.updateGraphs()
        self.qNetwork2.updateGraphs()
        self.learnedValueNetwork.updateGraphs()
        # self.targetValueNetwork.updateGraphs()
        self.policyNetwork.updateGraphs()

        self.timersPlot.cla()
        self.timersPlot.set_title("Timing")
        self.timersPlot.plot(self.getQTargetsOverTime, label="Q Targets")
        self.timersPlot.plot(self.getValueTargetsOverTime, label="Value Targets")
        self.timersPlot.plot(self.trainQOverTime, label="Train Q")
        self.timersPlot.plot(self.trainValueOverTime, label="Train Value")
        self.timersPlot.plot(self.trainPolicyOverTime, label="Train Policy")
        self.timersPlot.legend(loc=2)

        self.rewardsPlot.cla()
        self.rewardsPlot.set_title("Rewards")
        self.rewardsPlot.plot(self.rewardsOverTime, label="Rewards")
        plt.pause(0.00001)
    def execute(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.copyLearnedNetwork)
        self.globalStep = 0
        for episodeNum in range(self.maxEpisodes):
            if self.outOfTime():
                break
            self.state = self.env.reset()
            self.totalEpisodeReward = 0
            for stepNum in range(self.trainSteps):
                self.goToNextState()
            self.rewardsOverTime.append(self.totalEpisodeReward)
            if self.showGraphs:
                self.updateGraphs()
        self.state = self.env.reset()
        self.totalEpisodeReward = 0
        for stepNum in range(self.testSteps):
            self.goToNextState()
        return self.totalEpisodeReward
            


