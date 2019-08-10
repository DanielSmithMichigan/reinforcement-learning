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
from . import Actor

from .prioritized_experience_replay import PrioritizedExperienceReplay
from .simple_experience_replay import SimpleExperienceReplay

import threading

class Controller:
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
            saveModel,
            restoreModel,
            train,
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
            gradientSteps,
            initialExtraNoise,
            extraNoiseDecay,
            evaluationEvery,
            numFinalEvaluations,
            numThreads
        ):
        self.graph = tf.Graph()
        self.numStateVariables = 24
        self.numActions = 4
        self.batchSize = batchSize
        self.tau = tau
        self.extraNoiseMax = initialExtraNoise
        self.extraNoiseDecay = extraNoiseDecay
        self.evaluationEvery = evaluationEvery
        self.numFinalEvaluations = numFinalEvaluations
        self.restoreModel = restoreModel
        self.train = train
        with self.graph.as_default():
            self.sess = tf.Session()
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name="State_Placeholder")
            self.nextStatePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name="Next_State_Placeholder")
            self.actionsPh = tf.placeholder(tf.float32, [None, self.numActions], name="Actions_Placeholder")
            self.rewardsPh = tf.placeholder(tf.float32, [None, ], name="Rewards_Placeholder")
            self.terminalsPh = tf.placeholder(tf.float32, [None, ], name="Terminals_Placeholder")
            self.memoryPriorityPh = tf.placeholder(tf.float32, [None, ], name="MemoryPriority_Placeholder")
        self.trainingOperations = []
        self.env = gym.make('BipedalWalker-v2')
        self.startTime = time.time()
        self.randomStartSteps = randomStartSteps
        self.saveModel = saveModel
        self.gradientSteps = gradientSteps

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
            memoryPriorityPh=self.memoryPriorityPh,
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
            memoryPriorityPh=self.memoryPriorityPh,
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
            memoryPriorityPh=self.memoryPriorityPh,
            maxGradientNorm=maxGradientNorm
        )
        
        self.qNetwork2Target = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_2_Target_"+name,
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
            memoryPriorityPh=self.memoryPriorityPh,
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
        self.numThreads = numThreads
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
        self.actionsChosen = deque([], 400)
        self.q1Loss = deque([], 400)
        self.q2Loss = deque([], 400)
        self.entropyCoefficientLoss = deque([], 400)
        self.q1RegTerm = deque([], 400)
        self.q2RegTerm = deque([], 400)
        self.policyRegTerm = deque([], 400)
        self.entropyCoefficientOverTime = deque([], 400)
        self.entropyOverTime = deque([], 400)
        self.extraNoiseOverTime = deque([], 80000)
        self.lastGlobalStep = 0
        self.trainingSteps = 0
        self.lastTime = time.time()
        self.allThreads = []
        if showGraphs:
            self.buildGraphs()
        # self.logger = tf.summary.FileWriter('./log', graph_def=self.sess.graph_def)
    def buildGraphs(self):
        plt.ion()
        # self.buildAssessmentGraphs()

        self.overview = plt.figure()
        self.overview.suptitle("Overview")
        self.episodeRewardsGraph = self.overview.add_subplot(5, 1, 1)
        self.fpsOverTimeGraph = self.overview.add_subplot(5, 1, 2)
        self.actionsChosenGraph = self.overview.add_subplot(5, 1, 3)
        self.entropyOverTimeGraph = self.overview.add_subplot(5, 1, 4)
        self.extraNoiseOverTimeGraph = self.overview.add_subplot(5, 1, 5)
    
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

        self.extraNoiseOverTimeGraph.cla()
        self.extraNoiseOverTimeGraph.set_title("Extra Noise")
        self.extraNoiseOverTimeGraph.plot(self.extraNoiseOverTime, label="Extra Noise")

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
    def buildTrainingOperation(self):
        self.qNetwork1.setTargetNetworks(self.qNetwork1Target, self.qNetwork2Target)
        self.qNetwork1.setPolicyNetwork(self.policyNetwork)
        (
            qNetwork1Training,
            q1Loss,
            q1RegTerm,
            q1BatchwiseLoss
        ) = self.qNetwork1.buildTrainingOperation()

        self.qNetwork2.setTargetNetworks(self.qNetwork1Target, self.qNetwork2Target)
        self.qNetwork2.setPolicyNetwork(self.policyNetwork)
        (
            qNetwork2Training,
            q2Loss,
            q2RegTerm,
            q2BatchwiseLoss
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

        self.hardCopy1 = self.qNetwork1Target.buildSoftCopyOperation(self.qNetwork1, 1.0)
        self.hardCopy2 = self.qNetwork2Target.buildSoftCopyOperation(self.qNetwork2, 1.0)

        with self.graph.as_default():
            self.trainingOperations = [
                qNetwork1Training,
                qNetwork2Training,
                policyTrainingOperation,
                entropyCoefficientTrainingOperation,
                softCopy1,
                softCopy2,
                q1Loss,
                q1BatchwiseLoss,
                q1RegTerm,
                q2Loss,
                q2BatchwiseLoss,
                q2RegTerm,
                policyRegTerm,
                entropyCoefficientLoss,
                entropyCoefficient
                # tf.summary.merge_all()
            ]
    def trainNetworks(self):
        trainingMemories = self.memoryBuffer.getMemoryBatch()
        (
            qNetwork1Training,
            qNetwork2Training,
            policyTrainingOperation,
            entropyCoefficientTrainingOperation,
            softCopy1,
            softCopy2,
            q1Loss,
            q1BatchwiseLoss,
            q1RegTerm,
            q2Loss,
            q2BatchwiseLoss,
            q2RegTerm,
            policyRegTerm,
            entropyCoefficientLoss,
            entropyCoefficient
            # summary
        ) = self.sess.run(
            self.trainingOperations,
            feed_dict={
                self.statePh: util.getColumn(trainingMemories, constants.STATE),
                self.nextStatePh: util.getColumn(trainingMemories, constants.NEXT_STATE),
                self.actionsPh: util.getColumn(trainingMemories, constants.ACTION),
                self.terminalsPh: util.getColumn(trainingMemories, constants.IS_TERMINAL),
                self.rewardsPh: util.getColumn(trainingMemories, constants.REWARD),
                self.memoryPriorityPh: util.getColumn(trainingMemories, constants.PRIORITY)
            }
        )
        self.trainingSteps += 1
        self.q1Loss.append(q1Loss)
        self.q2Loss.append(q2Loss)
        self.q1RegTerm.append(q1RegTerm)
        self.q2RegTerm.append(q2RegTerm)
        self.policyRegTerm.append(policyRegTerm)
        self.entropyCoefficientLoss.append(entropyCoefficientLoss)
        self.entropyCoefficientOverTime.append(entropyCoefficient)
        for i in range(len(trainingMemories)):
            trainingMemories[i][constants.LOSS] = (q1BatchwiseLoss[i] + q2BatchwiseLoss[i]) / 2
        self.memoryBuffer.updateMemories(trainingMemories)
        # self.logger.add_summary(summary, self.globalStep)
    def updateFps(self):
        newTime = time.time()
        timeSpent = newTime - self.lastTime
        framesRendered = self.globalStep - self.lastGlobalStep 
        fps = framesRendered / timeSpent
        self.lastGlobalStep = self.globalStep
        self.lastTime = newTime
        self.fpsOverTime.append(fps)
        return fps
    def syncModelToS3(self):
        with self.graph.as_default():
            self.saver.save(self.sess, "./models/"+self.name)
        os.system("aws s3 sync models/ s3://tensorflow-models-dan-smith/"+self.name+"/checkpoint_"+str(self.checkpointNum)+"/")
        self.checkpointNum += 1
    def loadModel(self):
        with self.graph.as_default():
            self.saver.restore(self.sess, "./models/"+self.name)
    def execute(self):
        for i in range(self.numThreads):
            thread = threading.Thread(target=startThread, args=(self))

def startThread(controllerInstance):
    actor = Actor(
        policyNetwork=controllerInstance.policyNetwork,
        sess=controllerInstance.sess,
        memoryBuffer=controllerInstance.memoryBuffer,
        maxEpisodeSteps=controllerInstance.maxEpisodeSteps,
        envName=controllerInstance.envName,
        actionScaling=controllerInstance.actionScaling,
        deterministic=False
    )
    actor.execute()
            


