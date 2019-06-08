import tensorflow as tf
import numpy as np
import os
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
from . import constants
from . import util

from .prioritized_experience_replay import PrioritizedExperienceReplay
from .simple_experience_replay import SimpleExperienceReplay

class Agent:
    def __init__(self,
            name,
            policyNetworkSize,
            qNetworkSize,
            policyNetworkLearningRate,
            qNetworkLearningRate,
            entropyCoefficient,
            tau,
            gamma,
            maxMemoryLength,
            priorityExponent,
            batchSize,
            maxEpisodes,
            trainSteps,
            rewardScaling,
            stepsPerUpdate,
            render,
            showGraphs,
            syncToS3,
            clearBufferAfterTraining,
            minStepsBeforeTraining,
            actionScaling,
            actionShift,
            testSteps,
            maxMinutes,
            targetEntropy,
            maxGradientNorm,
            varianceRegularizationConstant,
            meanRegularizationConstant,
            randomStartSteps,
            gradientSteps
        ):
        self.graph = tf.Graph()
        self.numStateVariables = 24
        self.numActions = 4
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
        self.env = gym.make('BipedalWalker-v2')
        self.startTime = time.time()
        self.randomStartSteps = randomStartSteps
        self.syncToS3 = syncToS3
        self.gradientSteps = gradientSteps
        self.clearBufferAfterTraining = clearBufferAfterTraining

        self.qNetwork1 = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_1_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            showGraphs=showGraphs,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            rewardsPh=self.rewardsPh,
            terminalsPh=self.terminalsPh,
            maxGradientNorm=maxGradientNorm
        )


        self.qNetwork1Target = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_1_Target_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            showGraphs=showGraphs,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            rewardsPh=self.rewardsPh,
            terminalsPh=self.terminalsPh,
            maxGradientNorm=maxGradientNorm
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
            showGraphs=showGraphs,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            rewardsPh=self.rewardsPh,
            terminalsPh=self.terminalsPh,
            maxGradientNorm=maxGradientNorm
        )
        
        self.qNetwork2Target = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_2_Target"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            gamma=gamma,
            learningRate=qNetworkLearningRate,
            showGraphs=showGraphs,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            rewardsPh=self.rewardsPh,
            terminalsPh=self.terminalsPh,
            maxGradientNorm=maxGradientNorm
        )

        self.policyNetwork = PolicyNetwork(
            sess=self.sess,
            graph=self.graph,
            name="PolicyNetwork_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=policyNetworkSize,
            learningRate=policyNetworkLearningRate,
            batchSize=batchSize,
            showGraphs=showGraphs,
            statePh=self.statePh,
            targetEntropy=targetEntropy,
            entropyCoefficient=entropyCoefficient,
            maxGradientNorm=maxGradientNorm,
            varianceRegularizationConstant=varianceRegularizationConstant,
            meanRegularizationConstant=meanRegularizationConstant
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
        self.episodeRewards = deque([], 400)
        self.actionsChosen = deque([], 400)
        self.q1Loss = deque([], 400)
        self.q2Loss = deque([], 400)
        self.entropyCoefficientLoss = deque([], 400)
        self.q1RegTerm = deque([], 400)
        self.q2RegTerm = deque([], 400)
        self.policyRegTerm = deque([], 400)
        self.entropyCoefficientOverTime = deque([], 400)
        self.entropyOverTime = deque([], 400)
        self.lastGlobalStep = 0
        self.lastTime = time.time()
        if showGraphs:
            self.buildGraphs()
    def buildGraphs(self):
        plt.ion()
        # self.buildAssessmentGraphs()

        self.overview = plt.figure()
        self.overview.suptitle("Overview")
        self.episodeRewardsGraph = self.overview.add_subplot(4, 1, 1)
        self.fpsOverTimeGraph = self.overview.add_subplot(4, 1, 2)
        self.actionsChosenGraph = self.overview.add_subplot(4, 1, 3)
        self.entropyOverTimeGraph = self.overview.add_subplot(4, 1, 4)
    
        self.lossFigure = plt.figure()
        self.lossGraph = self.lossFigure.add_subplot(4, 1, 1)
        self.entropyCoefficientLossGraph = self.lossFigure.add_subplot(4, 1, 2)
        self.regTermGraph = self.lossFigure.add_subplot(4, 1, 3)
        self.entropyCoefficientGraph = self.lossFigure.add_subplot(4, 1, 4)
    def buildAssessmentGraphs(self):
        self.qAssessmentFigure = plt.figure()
        self.qAssessmentFigure.suptitle("Q Assessments")
        self.qAssessmentGraph = self.qAssessmentFigure.add_subplot(1, 1, 1)
        divider = make_axes_locatable(self.qAssessmentGraph)
        self.qAssessmentColorBar = divider.append_axes("right", size="7%", pad="2%")

        self.policyFigure = plt.figure()
        self.policyFigure.suptitle("Policy")
        self.policyGraph = self.policyFigure.add_subplot(1, 1, 1)
        divider = make_axes_locatable(self.policyGraph)
        self.policyColorBar = divider.append_axes("right", size="7%", pad="2%")
    def updateGraphs(self):
        # self.updateAssessmentGraphs()
        self.updateOverviewGraphs()
        self.updateLossGraphs()

        plt.pause(0.0001)
    def updateOverviewGraphs(self):

        self.episodeRewardsGraph.cla()
        self.episodeRewardsGraph.set_title("Rewards")
        self.episodeRewardsGraph.plot(self.episodeRewards, label="Rewards")

        self.fpsOverTimeGraph.cla()
        self.fpsOverTimeGraph.set_title("FPS")
        self.fpsOverTimeGraph.plot(self.fpsOverTime, label="FPS")

        self.actionsChosenGraph.cla()
        self.actionsChosenGraph.set_title("ActionsChosen")
        for i in range(self.numActions):
            self.actionsChosenGraph.plot(util.getColumn(self.actionsChosen, i), label=constants.ACTION_NAMES[i])
        self.actionsChosenGraph.legend(loc=2)

        self.entropyOverTimeGraph.cla()
        self.entropyOverTimeGraph.set_title("Entropy")
        self.entropyOverTimeGraph.plot(self.entropyOverTime, label="Entropy")

        self.overview.canvas.draw()
    def updateLossGraphs(self):
        self.lossGraph.cla()
        self.lossGraph.set_title("Loss")
        self.lossGraph.plot(self.q1Loss, label="q1")
        self.lossGraph.plot(self.q2Loss, label="q2")
        self.lossGraph.legend(loc=2)

        self.entropyCoefficientLossGraph.cla()
        self.entropyCoefficientLossGraph.set_title("Entropy Coefficient")
        self.entropyCoefficientLossGraph.plot(self.entropyCoefficientLoss, label="entropyCoefficient")
        self.entropyCoefficientLossGraph.legend(loc=2)

        self.regTermGraph.cla()
        self.regTermGraph.set_title("Reg Term")
        self.regTermGraph.plot(self.q1RegTerm, label="q1")
        self.regTermGraph.plot(self.q2RegTerm, label="q2")
        self.regTermGraph.plot(self.policyRegTerm, label="policy")
        self.regTermGraph.legend(loc=2)

        self.entropyCoefficientGraph.cla()
        self.entropyCoefficientGraph.set_title("Entropy Coefficient")
        self.entropyCoefficientGraph.plot(self.entropyCoefficientOverTime, label="Entropy Coefficient")

        self.lossFigure.canvas.draw()
    def updateAssessmentGraphs(self):
        states = []
        imageRadius = constants.IMAGE_SIZE / 2
        for xImg in range(constants.IMAGE_SIZE):
            for yImg in range(constants.IMAGE_SIZE):
                x = xImg - imageRadius
                y = yImg - imageRadius
                v = np.clip(math.sqrt(x * x + y * y) * 16 / imageRadius, 0, 16)
                v = v - 8

                theta = None
                if x < 0:
                    theta = math.atan(y / x) + math.pi
                elif x == 0 and y > 0:
                    theta = math.pi
                elif x == 0 and y < 0:
                    theta = -math.pi
                elif x == 0 and y == 0:
                    theta = 0
                elif x > 0 and y < 0:
                    theta = math.atan(y / x) + math.pi + math.pi
                else:
                    theta = math.atan(y / x)
                states.append([math.cos(theta), math.sin(theta), v])
        (
            actionsChosen,
            qAssessments
        ) = self.sess.run(
            self.graphingOperations,
            feed_dict={
                self.statePh: states
            }
        )
        actionsChosenImg = np.reshape(actionsChosen, [constants.IMAGE_SIZE, constants.IMAGE_SIZE])
        qAssessmentsImg = np.reshape(qAssessments, [constants.IMAGE_SIZE, constants.IMAGE_SIZE])

        self.qAssessmentGraph.cla()
        self.qAssessmentColorBar.cla()
        ax=self.qAssessmentGraph.imshow(qAssessmentsImg)
        colorbar(ax, cax=self.qAssessmentColorBar)
        self.qAssessmentFigure.canvas.draw()

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
                logProb,
                deterministicActionChosen
            ) = self.policyNetwork.buildNetwork(self.statePh)
            qAssessment = self.qNetwork1.buildNetwork(self.statePh, actionsChosen)
            self.actionOperations = [
                rawAction,
                actionsChosen,
                qAssessment,
                deterministicActionChosen,
                entropy
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
                logProb,
                deterministicActionChosen
            ) = self.policyNetwork.buildNetwork(self.statePh)
            qAssessment = self.qNetwork1.buildNetwork(self.statePh, actionsChosen)
            self.graphingOperations = [
                actionsChosen,
                qAssessment
            ]
    def goToNextState(self,deterministic=False):
        (
            rawAction,
            actionsChosen,
            qAssessment,
            deterministicAction,
            entropy
        ) = self.sess.run(
            self.actionOperations,
            feed_dict={
                self.statePh: [self.state]
            }
        )
        actionsChosen = actionsChosen[0] if not deterministic else deterministicAction[0]
        actionsChosen = actionsChosen * self.actionScaling
        if self.globalStep < self.randomStartSteps:
            actionsChosen = self.env.action_space.sample()
        self.actionsChosen.append(actionsChosen)
        self.entropyOverTime.append(entropy)
        nextState, reward, done, info = self.env.step(actionsChosen)
        nextState = np.reshape(nextState, [self.numStateVariables,])
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
            self.getLatestModel()
            for i in range(self.gradientSteps):
                self.train()
            self.updateModel()
        return done
    def buildTrainingOperation(self):
        self.qNetwork1.setTargetNetworks(self.qNetwork1Target, self.qNetwork2Target)
        self.qNetwork1.setPolicyNetwork(self.policyNetwork)
        (
            qNetwork1Training,
            q1Loss,
            q1RegTerm
        ) = self.qNetwork1.buildTrainingOperation()

        self.qNetwork2.setTargetNetworks(self.qNetwork1Target, self.qNetwork2Target)
        self.qNetwork2.setPolicyNetwork(self.policyNetwork)
        (
            qNetwork2Training,
            q2Loss,
            q2RegTerm
        ) = self.qNetwork2.buildTrainingOperation()

        self.policyNetwork.setQNetwork(self.qNetwork1)
        (
            policyTrainingOperation,
            entropyCoefficientTrainingOperation,
            policyRegTerm,
            entropyCoefficientLoss,
            entropyCoefficient
        ) = self.policyNetwork.buildTrainingOperation()

        softCopy1 = self.qNetwork1Target.buildSoftCopyOperation(self.qNetwork1, self.tau)
        softCopy2 = self.qNetwork2Target.buildSoftCopyOperation(self.qNetwork2, self.tau)

        self.trainingOperations = [
            qNetwork1Training,
            qNetwork2Training,
            policyTrainingOperation,
            entropyCoefficientTrainingOperation,
            softCopy1,
            softCopy2,
            q1Loss,
            q1RegTerm,
            q2Loss,
            q2RegTerm,
            policyRegTerm,
            entropyCoefficientLoss,
            entropyCoefficient
        ]

        self.hardCopy1 = self.qNetwork1Target.buildSoftCopyOperation(self.qNetwork1, 1.0)
        self.hardCopy2 = self.qNetwork2Target.buildSoftCopyOperation(self.qNetwork2, 1.0)
    def train(self):
        trainingMemories = self.memoryBuffer.getMemoryBatch()
        (
            qNetwork1Training,
            qNetwork2Training,
            policyTrainingOperation,
            entropyCoefficientTrainingOperation,
            softCopy1,
            softCopy2,
            q1Loss,
            q1RegTerm,
            q2Loss,
            q2RegTerm,
            policyRegTerm,
            entropyCoefficientLoss,
            entropyCoefficient
        ) = self.sess.run(
            self.trainingOperations,
            feed_dict={
                self.statePh: util.getColumn(trainingMemories, constants.STATE),
                self.nextStatePh: util.getColumn(trainingMemories, constants.NEXT_STATE),
                self.actionsPh: util.getColumn(trainingMemories, constants.ACTION),
                self.terminalsPh: util.getColumn(trainingMemories, constants.IS_TERMINAL),
                self.rewardsPh: util.getColumn(trainingMemories, constants.REWARD)
            }
        )
        self.q1Loss.append(q1Loss)
        self.q2Loss.append(q2Loss)
        self.q1RegTerm.append(q1RegTerm)
        self.q2RegTerm.append(q2RegTerm)
        self.policyRegTerm.append(policyRegTerm)
        self.entropyCoefficientLoss.append(entropyCoefficientLoss)
        self.entropyCoefficientOverTime.append(entropyCoefficient)
        if self.clearBufferAfterTraining:
            self.memoryBuffer.clear()
    def updateFps(self):
        newTime = time.time()
        timeSpent = newTime - self.lastTime
        framesRendered = self.globalStep - self.lastGlobalStep 
        fps = framesRendered / timeSpent
        self.lastGlobalStep = self.globalStep
        self.lastTime = newTime
        self.fpsOverTime.append(fps)
        return fps
    def getLatestModel(self):
        if self.syncToS3:
            os.system("aws s3 sync s3://tensorflow-models-dan-smith/"+self.name+"/ models/")
            if tf.train.checkpoint_exists("./models/"+self.name):
                with self.graph.as_default():
                    self.saver.restore(self.sess, "./models/"+self.name)
    def updateModel(self):
        if self.syncToS3:
            with self.graph.as_default():
                self.saver.save(self.sess, "./models/"+self.name)
            os.system("aws s3 sync models/ s3://tensorflow-models-dan-smith/"+self.name+"/")
    def execute(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            if self.syncToS3:
                self.saver = tf.train.Saver()
        self.sess.run([self.hardCopy1, self.hardCopy2])
        self.globalStep = 0
        for episodeNum in range(self.maxEpisodes):
            if self.outOfTime():
                break
            state = self.env.reset()
            self.state = np.reshape(state, [self.numStateVariables,])
            self.totalEpisodeReward = 0
            for stepNum in range(self.trainSteps):
                done = self.goToNextState()
                if done:
                    break
            self.episodeRewards.append(self.totalEpisodeReward)
            fps = self.updateFps()
            print("REWARD: "+str(self.totalEpisodeReward)+" FPS: "+str(fps))
            if self.showGraphs:
                self.updateGraphs()
            self.getLatestModel()
        state = self.env.reset()
        self.state = np.reshape(state, [self.numStateVariables,])
        self.totalEpisodeReward = 0
        for stepNum in range(self.testSteps):
            done = self.goToNextState(deterministic=True)
            if done:
                break
        return self.totalEpisodeReward
            


