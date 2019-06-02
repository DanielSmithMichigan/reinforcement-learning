import tensorflow as tf
import numpy as np
import gym
import time
import math
from collections import deque
import multiprocessing
import sys

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import time

from .PolicyNetwork import PolicyNetwork
from .QNetwork import QNetwork
from .ValueNetwork import ValueNetwork
from . import constants
from . import util

from .prioritized_experience_replay import PrioritizedExperienceReplay
from .simple_experience_replay import SimpleExperienceReplay


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
            maxMinutes,
            theta,
            sigma,
            epsilonDecay,
            epsilonInitial
        ):
        self.graph = tf.Graph()
        self.numStateVariables = 3
        self.numActions = 1
        self.entropyCoefficient = entropyCoefficient
        self.batchSize = batchSize
        self.tau = tau
        with self.graph.as_default():
            self.sess = tf.Session()
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name="State_Placeholder")
            self.nextStatePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name="Next_State_Placeholder")
            self.actionsPh = tf.placeholder(tf.float32, [None, self.numActions], name="Actions_Placeholder")
            self.rewardsPh = tf.placeholder(tf.float32, [None, ], name="Rewards_Placeholder")
            self.terminalsPh = tf.placeholder(tf.float32, [None, ], name="Terminals_Placeholder")
        self.trainingOperations = []
        self.env = gym.make('Pendulum-v0')
        self.startTime = time.time()

        self.learnedValueNetwork = ValueNetwork(
            sess=self.sess,
            graph=self.graph,
            name="LearnedValueNetwork_"+name,
            numStateVariables=self.numStateVariables,
            networkSize=valueNetworkSize,
            entropyCoefficient=entropyCoefficient,
            learningRate=valueNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            numActions=self.numActions,
            showGraphs=showGraphs,
            statePh=self.statePh
        )
        self.targetValueNetwork = ValueNetwork(
            name="TargetValueNetwork_"+name,
            sess=self.sess,
            graph=self.graph,
            numStateVariables=self.numStateVariables,
            networkSize=valueNetworkSize,
            entropyCoefficient=entropyCoefficient,
            learningRate=valueNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize,
            numActions=self.numActions,
            showGraphs=showGraphs,
            statePh=self.statePh
        )

        self.qNetwork1 = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_1_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            showGraphs=showGraphs,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            rewardsPh=self.rewardsPh,
            terminalsPh=self.terminalsPh
        )
        self.qNetwork2 = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_2_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            maxGradientNorm=maxGradientNorm,
            showGraphs=showGraphs,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            rewardsPh=self.rewardsPh,
            terminalsPh=self.terminalsPh
        )

        self.policyNetwork = PolicyNetwork(
            sess=self.sess,
            graph=self.graph,
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
            showGraphs=showGraphs,
            statePh=self.statePh
        )

        self.buildTrainingOperation()
        self.buildActionOperation()
        self.buildGraphingOperation()

        if float(priorityExponent) != 0.0:
            self.memoryBuffer = PrioritizedExperienceReplay(
                numMemories=maxMemoryLength,
                priorityExponent=priorityExponent,
                batchSize=batchSize
            )
        else:
            self.memoryBuffer = SimpleExperienceReplay(
                numMemories=maxMemoryLength,
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
        self.fpsOverTime = deque([], 400)
        self.lastGlobalStep = 0
        self.lastTime = time.time()
        if showGraphs:
            self.buildGraphs()
    def buildGraphs(self):
        plt.ion()
        self.qAssessmentFigure = plt.figure()
        self.qAssessmentFigure.suptitle("Q Assessments")
        self.qAssessmentGraph = self.qAssessmentFigure.add_subplot(1, 1, 1)
        divider = make_axes_locatable(self.qAssessmentGraph)
        self.qAssessmentColorBar = divider.append_axes("right", size="7%", pad="2%")

        self.valueAssessmentFigure = plt.figure()
        self.valueAssessmentFigure.suptitle("Value Assessments")
        self.valueAssessmentGraph = self.valueAssessmentFigure.add_subplot(1, 1, 1)
        divider = make_axes_locatable(self.valueAssessmentGraph)
        self.valueAssessmentColorBar = divider.append_axes("right", size="7%", pad="2%")

        self.policyFigure = plt.figure()
        self.policyFigure.suptitle("Policy")
        self.policyGraph = self.policyFigure.add_subplot(1, 1, 1)
        divider = make_axes_locatable(self.policyGraph)
        self.policyColorBar = divider.append_axes("right", size="7%", pad="2%")
    def updateGraphs(self):
        self.updateEvalGraphs()

        plt.pause(0.0001)
    def updateEvalGraphs(self):
        states = []
        imageRadius = constants.IMAGE_SIZE / 2
        for xImg in range(constants.IMAGE_SIZE):
            for yImg in range(constants.IMAGE_SIZE):
                x = xImg - imageRadius
                y = yImg - imageRadius
                v = np.clip(math.sqrt(x * x + y * y) * 16 / imageRadius, 0, 16)
                v = v - 8
                theta = math.atan(y / x) if x != 0 else math.pi / 2
                if x < 0 and y > 0:
                    theta = theta + math.pi
                elif x < 0 and y < 0:
                    theta = theta + math.pi
                elif x > 0 and y < 0:
                    theta = theta + math.pi + math.pi
                states.append([math.cos(theta), math.sin(theta), v])
        (
            actionsChosen,
            qAssessments,
            valueAssessments
        ) = self.sess.run(
            self.graphingOperations,
            feed_dict={
                self.statePh: states
            }
        )
        actionsChosenImg = np.reshape(actionsChosen, [constants.IMAGE_SIZE, constants.IMAGE_SIZE])
        qAssessmentsImg = np.reshape(qAssessments, [constants.IMAGE_SIZE, constants.IMAGE_SIZE])
        valueAssessmentsImg = np.reshape(valueAssessments, [constants.IMAGE_SIZE, constants.IMAGE_SIZE])

        self.qAssessmentGraph.cla()
        self.qAssessmentColorBar.cla()
        ax=self.qAssessmentGraph.imshow(qAssessmentsImg)
        colorbar(ax, cax=self.qAssessmentColorBar)
        self.qAssessmentFigure.canvas.draw()

        self.valueAssessmentGraph.cla()
        self.valueAssessmentColorBar.cla()
        ax=self.valueAssessmentGraph.imshow(valueAssessmentsImg)
        colorbar(ax, cax=self.valueAssessmentColorBar)
        self.valueAssessmentFigure.canvas.draw()

        self.policyGraph.cla()
        self.policyColorBar.cla()
        ax=self.policyGraph.imshow(actionsChosenImg)
        colorbar(ax, cax=self.policyColorBar)
        self.policyFigure.canvas.draw()
    def outOfTime(self):
        return time.time() > self.startTime + (self.maxMinutes * 60)
    def buildActionOperation(self):
        with self.graph.as_default():
            (
                actionsChosen,
                rawAction,
                actionMean,
                uncleanedActionVariance,
                logScaleActionVariance,
                actionVariance,
                entropy,
                logProb
            ) = self.policyNetwork.buildNetwork(self.statePh)
            qAssessment = self.qNetwork1.buildNetwork(self.statePh, actionsChosen)
            self.actionOperations = [
                actionsChosen,
                qAssessment
            ]
    def buildGraphingOperation(self):
        with self.graph.as_default():
            (
                actionsChosen,
                rawAction,
                actionMean,
                uncleanedActionVariance,
                logScaleActionVariance,
                actionVariance,
                entropy,
                logProb
            ) = self.policyNetwork.buildNetwork(self.statePh)
            qAssessment = self.qNetwork1.buildNetwork(self.statePh, actionsChosen)
            valueAssessment = self.learnedValueNetwork.buildNetwork(self.statePh)
            self.graphingOperations = [
                actionsChosen,
                qAssessment,
                valueAssessment
            ]
    def goToNextState(self):
        (
            actionsChosen,
            qAssessment
        ) = self.sess.run(
            self.actionOperations,
            feed_dict={
                self.statePh: [self.state]
            }
        )
        actionsChosen = actionsChosen[0]
        nextState, reward, done, info = self.env.step(actionsChosen * self.actionScaling)
        nextState = np.reshape(nextState, [self.numStateVariables,])
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
    def buildTrainingOperation(self):
        self.qNetwork1.setValueNetwork(self.targetValueNetwork)
        self.qNetwork1.setPolicyNetwork(self.policyNetwork)
        (
            qNetwork1Training
        ) = self.qNetwork1.buildTrainingOperation()

        self.qNetwork2.setValueNetwork(self.targetValueNetwork)
        self.qNetwork2.setPolicyNetwork(self.policyNetwork)
        (
            qNetwork2Training
        ) = self.qNetwork2.buildTrainingOperation()

        self.learnedValueNetwork.setNetworks(self.policyNetwork, self.qNetwork1, self.qNetwork2)
        (
            learnedValueTraining
        ) = self.learnedValueNetwork.buildTrainingOperation()

        self.targetValueNetwork.setNetworks(self.policyNetwork, self.qNetwork1, self.qNetwork2)

        self.policyNetwork.setQNetwork(self.qNetwork1)
        (
            policyTraining
        ) = self.policyNetwork.buildTrainingOperation()

        softCopy = self.targetValueNetwork.buildSoftCopyOperation(self.learnedValueNetwork, self.tau)

        self.trainingOperations = [
            qNetwork1Training,
            qNetwork2Training,
            learnedValueTraining,
            policyTraining,
            softCopy
        ]

        self.copyLearnedNetwork = self.targetValueNetwork.buildSoftCopyOperation(self.learnedValueNetwork, 1)
    def train(self):
        trainingMemories = self.memoryBuffer.getMemoryBatch()
        (
            qNetwork1Training,
            qNetwork2Training,
            learnedValueTraining,
            policyTraining,
            softCopy
        ) = self.sess.run(
            self.trainingOperations,
            feed_dict={
                self.statePh: util.getColumn(trainingMemories, constants.STATE),
                self.nextStatePh: util.getColumn(trainingMemories, constants.NEXT_STATE),
                self.actionsPh: util.getColumn(trainingMemories, constants.ACTION),
                self.terminalsPh: util.getColumn(trainingMemories, constants.IS_TERMINAL),
                self.rewardsPh: util.getColumn(trainingMemories, constants.REWARD),
                self.policyNetwork.entropyCoefficientPh: self.entropyCoefficient
            }
        )
    def updateFps(self):
        newTime = time.time()
        timeSpent = newTime - self.lastTime
        framesRendered = self.globalStep - self.lastGlobalStep 
        fps = framesRendered / timeSpent
        self.lastGlobalStep = self.globalStep
        self.lastTime = newTime
        self.fpsOverTime.append(fps)
        return fps
    def execute(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.copyLearnedNetwork)
        self.globalStep = 0
        for episodeNum in range(self.maxEpisodes):
            if self.outOfTime():
                break
            state = self.env.reset()
            self.state = np.reshape(state, [self.numStateVariables,])
            self.totalEpisodeReward = 0
            for stepNum in range(self.trainSteps):
                self.goToNextState()
            fps = self.updateFps()
            if self.showGraphs:
                self.updateGraphs()
            # print("Reward: "+str(self.totalEpisodeReward)+" FPS: "+str(fps))
        state = self.env.reset()
        self.state = np.reshape(state, [self.numStateVariables,])
        self.totalEpisodeReward = 0
        for stepNum in range(self.testSteps):
            self.goToNextState()
        return self.totalEpisodeReward
            


