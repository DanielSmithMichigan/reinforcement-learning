import tensorflow as tf
import numpy as np
import gym

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
            maxSteps,
            rewardScaling,
            stepsPerUpdate,
            render,
            minStepsBeforeTraining,
            actionScaling,
            theta,
            sigma
        ):
        self.numStateVariables = 3
        self.numActions = 1
        self.env = gym.make('Pendulum-v0')
        self.sess = tf.Session()

        self.learnedValueNetwork = ValueNetwork(
            sess=self.sess,
            name="LearnedValueNetwork_"+name,
            numStateVariables=self.numStateVariables,
            networkSize=valueNetworkSize,
            entropyCoefficient=entropyCoefficient,
            learningRate=valueNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            numActions=self.numActions
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
            numActions=self.numActions
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
            maxGradientNorm=maxGradientNorm
        )
        self.qNetwork2 = QNetwork(
            sess=self.sess,
            name="QNetwork_2_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            maxGradientNorm=maxGradientNorm
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
            theta=theta,
            sigma=sigma
        )

        self.qNetwork1.setValueNetwork(self.targetValueNetwork)
        self.qNetwork1.buildTrainingOperation()

        self.qNetwork2.setValueNetwork(self.targetValueNetwork)
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
        self.maxSteps = maxSteps
        self.rewardScaling = rewardScaling
        self.gamma = gamma
        self.stepsPerUpdate = stepsPerUpdate
        self.render = render
        self.minStepsBeforeTraining = minStepsBeforeTraining
        self.actionScaling = actionScaling
    def goToNextState(self):
        actionsChosen = self.policyNetwork.getAction(self.state)
        self.qNetwork1.storeAssessment(self.state, actionsChosen)
        self.qNetwork2.storeAssessment(self.state, actionsChosen)
        nextState, reward, done, info = self.env.step(actionsChosen * self.actionScaling)
        if (self.render):
            self.env.render()
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = self.state
        memoryEntry[constants.ACTION] = actionsChosen    
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
        qOneTargets = self.qNetwork1.getTargets(trainingMemories)
        qTwoTargets = self.qNetwork2.getTargets(trainingMemories)
        valueTargets = self.targetValueNetwork.getTargets(trainingMemories)
        self.qNetwork1.trainAgainst(trainingMemories, qOneTargets)
        self.qNetwork2.trainAgainst(trainingMemories, qTwoTargets)
        self.learnedValueNetwork.trainAgainst(trainingMemories, valueTargets)
        self.policyNetwork.trainAgainst(trainingMemories)
        self.sess.run(self.softCopyLearnedNetwork)
    def updateGraphs(self):
        self.qNetwork1.updateGraphs()
        self.qNetwork2.updateGraphs()
        self.learnedValueNetwork.updateGraphs()
        self.policyNetwork.updateGraphs()
    def execute(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.copyLearnedNetwork)
        self.globalStep = 0
        for episodeNum in range(self.maxEpisodes):
            self.state = self.env.reset()
            self.totalEpisodeReward = 0
            for stepNum in range(self.maxSteps):
                done = self.goToNextState()
                # print("Done: ",done)
                # if done:
                #     break
            print("Episode Reward: ",self.totalEpisodeReward)
            self.updateGraphs()


