from collections import deque
from .network import Network
from .buffer import Buffer
from . import constants
from . import util
from .prioritized_experience_replay import PrioritizedExperienceReplay
from .memoryBatcher import MemoryBatcher
import numpy as np
import tensorflow as tf
import random
import matplotlib
import matplotlib.pyplot as plt
import time
plt.ion()

def toList(d):
    output = []
    for i in range(len(d)):
        output.append(d[i])
    return output

#TODO: Update learned network periodically + tests
#TODO: Tests on execution function
#TODO: Re-read DQN paper

class Agent:
    def __init__(
            self,
            sess,
            env,
            maxMemoryLength,
            gamma,
            batchSize,
            numAvailableActions,
            numObservations,
            learningRate,
            episodeStepLimit,
            nStepUpdate,
            priorityExponent,
            minFramesForTraining,
            includeIntermediatePairs,
            render,
            showGraph,
            nStepReturns,
            epsilonInitial,
            epsilonDecay,
            numTestPeriods,
            numTestsPerTestPeriod,
            episodesPerTest,
            stepsPerTrainingPeriod,
            intermediateTests,
            rewardsMovingAverageSampleLength,
            maxGradientNorm,
            minExploration,
            maxExploration,
            maxRunningMinutes,
            preNetworkSize,
            postNetworkSize,
            numQuantiles,
            embeddingDimension,
            kappa,
            tau,
            trainingIterations,
            saveModel,
            loadModel,
            disableRandomActions,
            disableTraining,
            rewardScaling,
            agentName="agent_" + str(np.random.randint(low=100000000, high=999999999))
        ):
        self.agentName = agentName
        self.startTime = time.time()
        self.sess = sess
        self.env = env
        self.batchSize = batchSize
        self.numQuantiles = numQuantiles
        self.trainingIterations = trainingIterations
        self.epsilonDecay = epsilonDecay
        self.epsilon = epsilonInitial
        self.minExploration = minExploration
        self.maxExploration = maxExploration
        self.episodesPerTest = episodesPerTest
        self.numAvailableActions = numAvailableActions
        self.maxRunningMinutes = maxRunningMinutes
        self.stepsPerTrainingPeriod = stepsPerTrainingPeriod
        self.gamma = gamma
        self.tau = tau
        self.episodeStepLimit = episodeStepLimit
        self.totalEpisodeReward = 0
        self.globalStep = 0
        self.maxMeanReward = -10000000
        self.render = render
        self.numTestPeriods = numTestPeriods
        self.numTestsPerTestPeriod = numTestsPerTestPeriod
        self.showGraph = showGraph
        self.saveModel = saveModel
        self.loadModel = loadModel
        self.rewardScaling = rewardScaling
        self.disableRandomActions = disableRandomActions
        self.disableTraining = disableTraining
        self.numTestsPerTestPeriod = numTestsPerTestPeriod
        self.memoryBatcher = MemoryBatcher(
            nStepReturns=nStepReturns
        )
        self.memoryBuffer = PrioritizedExperienceReplay(
            numMemories=maxMemoryLength,
            priorityExponent=priorityExponent,
            batchSize=batchSize
        )
        self.targetNetwork = Network(
            name="target-network-" + self.agentName,
            sess=sess,
            showGraph=False,
            numObservations=numObservations,
            numAvailableActions=numAvailableActions,
            learningRate=learningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            preNetworkSize=preNetworkSize,
            postNetworkSize=postNetworkSize,
            numQuantiles=numQuantiles,
            embeddingDimension=embeddingDimension,
            kappa=kappa
        )
        self.learnedNetwork = Network(
            name="learned-network-" + self.agentName,
            sess=sess,
            showGraph=False,
            numObservations=numObservations,
            numAvailableActions=numAvailableActions,
            learningRate=learningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            preNetworkSize=preNetworkSize,
            postNetworkSize=postNetworkSize,
            numQuantiles=numQuantiles,
            embeddingDimension=embeddingDimension,
            kappa=kappa,
            targetNetwork=self.targetNetwork
        )
        self.duplicateLearnedNetwork = self.targetNetwork.buildSoftCopyOperation(self.learnedNetwork.networkParams, 1)
        self.softCopyLearnedNetwork = self.targetNetwork.buildSoftCopyOperation(self.learnedNetwork.networkParams, self.tau)
        self.minFramesForTraining = minFramesForTraining
        self.intermediateTests = intermediateTests
        self.testOutput = []
        self.rewardsMovingAverageSampleLength = rewardsMovingAverageSampleLength
        if showGraph:
            self.buildGraphs()
        #Build placeholder graph values
        self.targetExample = 0
        self.actualExample = 0
        self.actionExample = 0
        self.recentTarget = 0
        self.recentPrediction = 0
        self.rewardsMovingAverage = []
        self.rewardsReceivedOverTime = []
        self.rewardsStdDev = []
        self.agentAssessmentsOverTime = []
        self.epsilonOverTime = []
        self.choicesOverTime = []
        self.saver = tf.train.Saver()
    def getAgentAssessment(self, state):
        qValues, maxQ = self.sess.run([
            self.learnedNetwork.qValues,
            self.learnedNetwork.maxQ
        ], feed_dict={
            self.learnedNetwork.environmentInput: [state],
            self.learnedNetwork.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(1, self.numQuantiles))
        })
        self.agentAssessmentsOverTime.append(maxQ[0])
        self.recentAgentQValues = qValues[0]
    def getAction(self, state):
        return self.learnedNetwork.getAction(state)
    def goToNextState(self, currentState, actionChosen, stepNum):
        nextState, reward, done, info = self.env.step(actionChosen)
        self.totalEpisodeReward = self.totalEpisodeReward + reward
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = currentState
        memoryEntry[constants.ACTION] = actionChosen
        memoryEntry[constants.REWARD] = reward * self.rewardScaling
        memoryEntry[constants.NEXT_STATE] = nextState
        memoryEntry[constants.GAMMA] = self.gamma if not done else 0
        memoryEntry[constants.IS_TERMINAL] = done
        self.episodeMemories.append(memoryEntry)
        return nextState, done
    def train(self):
        trainingEpisodes = self.memoryBuffer.getMemoryBatch()
        targets, predictions, actions = self.learnedNetwork.trainAgainst(trainingEpisodes)
        choice = np.random.randint(len(targets))
        choice2 = np.random.randint(len(targets[choice]))
        choice3 = np.random.randint(len(targets[choice][choice2]))
        self.sess.run(self.softCopyLearnedNetwork)
        self.recentTarget = targets[choice][choice2][choice3]
        self.recentPrediction = predictions[choice][choice2][choice3]
        self.recentAction = actions[choice]
        self.memoryBuffer.updateMemories(trainingEpisodes)
    def buildGraphs(self):
        self.overview = plt.figure()
        self.lastNRewardsGraph = self.overview.add_subplot(4, 1, 1)

        self.rewardsReceivedOverTimeGraph = self.overview.add_subplot(4, 2, 3)
        self.rewardsReceivedOverTimeGraph.set_ylabel('Reward')
        self.rewardsReceivedOverTimeGraph.set_xlabel('Episode #')

        self.lossesGraph = self.overview.add_subplot(4, 2, 4)
        self.lossesGraph.set_ylabel('Loss amount')
        self.lossesGraph.set_xlabel('Iteration')

        self.agentAssessmentGraph = self.overview.add_subplot(4, 2, 5)
        self.agentAssessmentGraph.set_ylabel('Expected Value')
        self.agentAssessmentGraph.set_xlabel('Episode #')

        self.epsilonOverTimeGraph = self.overview.add_subplot(4, 2, 6)
        self.epsilonOverTimeGraph.set_ylabel('Epsilon')
        self.epsilonOverTimeGraph.set_xlabel('Episode #')

        self.choicesOverTimeGraph = self.overview.add_subplot(4, 1, 4)

        self.qValuesFigure = plt.figure()

        self.recentTrainingExampleGraph = self.qValuesFigure.add_subplot(2, 1, 1)

        self.qValueExample = self.qValuesFigure.add_subplot(2, 1, 2)
        self.qValueExample.set_ylabel('Probability')
        self.qValueExample.set_xlabel('Value')
        self.qValueExample.set_title("Prediction")

        self.recentAction = 0
    def updateGraphs(self):
        self.lastNRewardsGraph.cla()
        self.lastNRewardsGraph.plot(self.rewardsReceivedOverTime[-self.rewardsMovingAverageSampleLength:], label="Reward")
        self.lastNRewardsGraph.plot(self.rewardsMovingAverage[-self.rewardsMovingAverageSampleLength:], label="Moving average")
        self.rewardsReceivedOverTimeGraph.cla()
        self.rewardsReceivedOverTimeGraph.plot(self.rewardsReceivedOverTime)
        self.lossesGraph.cla()
        self.lossesGraph.plot(self.learnedNetwork.losses)
        self.agentAssessmentGraph.cla()
        self.agentAssessmentGraph.plot(self.agentAssessmentsOverTime)
        self.epsilonOverTimeGraph.cla()
        self.epsilonOverTimeGraph.plot(self.epsilonOverTime)
        self.choicesOverTimeGraph.cla()
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 0), label=constants.ACTION_NAMES[0])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 1), label=constants.ACTION_NAMES[1])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 2), label=constants.ACTION_NAMES[2])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 3), label=constants.ACTION_NAMES[3])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 4), label="Epsilon", linestyle=":")
        self.choicesOverTimeGraph.legend(loc=2)
        self.overview.canvas.draw()

        self.recentTrainingExampleGraph.cla()
        self.recentTrainingExampleGraph.bar(["Target", "Prediction"], [self.recentTarget, self.recentPrediction])
        self.recentTrainingExampleGraph.set_ylabel("Q Value")
        self.recentTrainingExampleGraph.set_title(constants.ACTION_NAMES[self.recentAction])
        self.qValueExample.cla()
        self.qValueExample.bar(constants.ACTION_NAMES, self.recentAgentQValues)
        self.qValueExample.set_ylabel("Q Value")
        self.qValuesFigure.canvas.draw()

        # self.learnedNetwork.updateGraphs()
        plt.pause(0.00001)
    def playEpisode(self, useRandomActions, recordTestResult, testNum=0):
        epsilon = 0
        if useRandomActions and (not self.disableRandomActions):
            self.epsilon = self.epsilon * self.epsilonDecay
            epsilon = min(1 * self.epsilon, 1)
            epsilon = max(epsilon, self.minExploration)
            epsilon = min(epsilon, self.maxExploration)
        self.epsilonOverTime.append(self.epsilon)
        state = self.env.reset()
        self.getAgentAssessment(state)
        self.totalEpisodeReward = 0
        stepNum = 0
        self.episodeMemories = []
        agentChoices = np.zeros(self.numAvailableActions + 1)
        while True:
            self.globalStep += 1
            if (self.globalStep % self.stepsPerTrainingPeriod) == 0:
                if self.memoryBuffer.length > self.minFramesForTraining and not (self.disableTraining):
                    self.train()
            stepNum = stepNum + 1
            if np.random.random() > epsilon:
                actionChosen = self.getAction(state)
            else:
                actionChosen = np.random.randint(self.numAvailableActions)
            agentChoices[actionChosen] = agentChoices[actionChosen] + 1
            state, done = self.goToNextState(state, actionChosen, stepNum)
            if self.render:
                self.env.render()
            if stepNum > self.episodeStepLimit:
                break
            if done:
                break
        self.rewardsReceivedOverTime.append(self.totalEpisodeReward)
        meanReward = np.mean(self.rewardsReceivedOverTime[-self.rewardsMovingAverageSampleLength:])
        print(self.agentName + " Episode ",len(self.rewardsReceivedOverTime),": ",meanReward)
        if self.saveModel:
            if meanReward > self.maxMeanReward:
                self.maxMeanReward = meanReward
                self.saver.save(self.sess, "models/"+self.agentName+".ckpt")
        periodResults = self.rewardsReceivedOverTime[-self.numTestsPerTestPeriod:]
        mu = np.mean(periodResults)
        std = np.std(periodResults)
        self.rewardsMovingAverage.append(mu)
        self.rewardsStdDev.append(std)
        sumChoices = np.sum(agentChoices)
        agentChoices = agentChoices / sumChoices
        agentChoices[self.numAvailableActions] = epsilon
        self.choicesOverTime.append(agentChoices)
        # batchedEpisodeMemories = self.memoryBatcher.batch(self.episodeMemories)
        for i in self.episodeMemories:
            self.memoryBuffer.add(i)
        if (recordTestResult):
            self.testResults.append(self.totalEpisodeReward)
    def executeTestPeriod(self):
        if (self.numTestsPerTestPeriod > 0):
            self.testResults = []
            for test_num in range(self.numTestsPerTestPeriod):
                self.playEpisode(useRandomActions=False,recordTestResult=True,testNum=test_num)
            print("Agent: " + self.agentName + " Test "+str(len(self.testResults))+": "+str(np.mean(self.testResults)))
            self.testOutput.append(np.mean(self.testResults))
    def outOfTime(self):
        return time.time() > self.startTime + (self.maxRunningMinutes * 60)
    def initModelVariables(self):
        if (self.loadModel):
            self.saver.restore(self.sess, "models/"+self.agentName+".ckpt")
        else:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.duplicateLearnedNetwork)
    def execute(self):
        self.initModelVariables()
        for testNum in range(self.numTestPeriods):
            for episodeNum in range(self.episodesPerTest):
                self.playEpisode(useRandomActions=True,recordTestResult=False)
                if self.outOfTime():
                    break
            if self.showGraph:
                self.updateGraphs()
            if self.intermediateTests:
                self.executeTestPeriod()
            if self.outOfTime():
                break
        self.executeTestPeriod()
        return self.testOutput