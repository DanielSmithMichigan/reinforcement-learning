import tensorflow as tf
import numpy as np
import os
import gym
import time
import math
from collections import deque
import multiprocessing
import sys
import threading

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import time

from .PolicyNetwork import PolicyNetwork
from .QNetwork import QNetwork
from .Actor import Actor
from .Trainer import Trainer
from . import constants
from . import util

from .prioritized_experience_replay import PrioritizedExperienceReplay
from .simple_experience_replay import SimpleExperienceReplay

class Agent:
    def __init__(self,
            name,
            numStateVariables,
            numActionVariables,
            envName,
            policyNetworkSize,
            qNetworkSize,
            numQuantiles,
            policyNetworkLearningRate,
            qNetworkLearningRate,
            entropyCoefficient,
            tau,
            gamma,
            kappa,
            maxMemoryLength,
            priorityExponent,
            batchSize,
            maxEpisodes,
            maxTrainSteps,
            trainSteps,
            rewardScaling,
            stepsPerUpdate,
            render,
            showGraphs,
            saveModel,
            saveModelToS3,
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
            numFinalEvaluations
        ):
        self.graph = tf.Graph()
        self.numStateVariables = numStateVariables
        self.numActions = numActionVariables
        self.envName = envName
        self.batchSize = batchSize
        self.tau = tau
        self.extraNoiseMax = initialExtraNoise
        self.extraNoiseDecay = extraNoiseDecay
        self.evaluationEvery = evaluationEvery
        self.numFinalEvaluations = numFinalEvaluations
        self.restoreModel = restoreModel
        self.train = train
        self.trainingOperations = []
        self.actors = []
        self.startTime = time.time()
        self.randomStartSteps = randomStartSteps
        self.saveModel = saveModel
        self.saveModelToS3 = saveModelToS3
        self.gradientSteps = gradientSteps

        self.name = name
        self.maxEpisodes = maxEpisodes
        self.maxTrainSteps = maxTrainSteps
        self.trainSteps = trainSteps
        self.rewardScaling = rewardScaling
        self.numQuantiles = numQuantiles
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
        self.extraNoiseOverTime = deque([], 80000)
        self.lastGlobalStep = 0
        self.trainingSteps = 0
        self.lastTime = time.time()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.statePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name="State_Placeholder")
            self.nextStatePh = tf.placeholder(tf.float32, [None, self.numStateVariables], name="Next_State_Placeholder")
            self.actionsPh = tf.placeholder(tf.float32, [None, self.numActions], name="Actions_Placeholder")
            self.rewardsPh = tf.placeholder(tf.float32, [None, ], name="Rewards_Placeholder")
            self.terminalsPh = tf.placeholder(tf.float32, [None, ], name="Terminals_Placeholder")
            self.memoryPriorityPh = tf.placeholder(tf.float32, [None, ], name="MemoryPriority_Placeholder")

        self.qNetwork1 = QNetwork(
            sess=self.sess,
            graph=self.graph,
            name="QNetwork_1_"+name,
            numStateVariables=self.numStateVariables,
            numActions=self.numActions,
            networkSize=qNetworkSize,
            numQuantiles=numQuantiles,
            gamma=gamma,
            kappa=kappa,
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
            numQuantiles=numQuantiles,
            gamma=gamma,
            kappa=kappa,
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

        self.createActors()
        self.trainer = Trainer(
            sess=self.sess,
            trainingOperations=self.trainingOperations,
            memoryBuffer=self.memoryBuffer,
            statePh=self.statePh,
            nextStatePh=self.nextStatePh,
            actionsPh=self.actionsPh,
            terminalsPh=self.terminalsPh,
            rewardsPh=self.rewardsPh,
            actors=self.actors,
            minStepsBeforeTraining=minStepsBeforeTraining
        )
        # with self.graph.as_default():
            # self.logger = tf.summary.FileWriter('./log', graph_def=self.sess.graph_def)
        if showGraphs:
            self.buildGraphs()
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
        return (time.time() > self.startTime + (self.maxMinutes * 60)) or (self.globalStep >= self.maxTrainSteps)
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
            (
                _,
                qAssessment
            ) = self.qNetwork1.buildNetwork(self.statePh, actionsChosen)
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
            (
                _,
                qAssessment
            ) = self.qNetwork1.buildNetwork(self.statePh, actionsChosen)
            self.graphingOperations = [
                actionsChosen,
                qAssessment
            ]
    def buildTrainingOperation(self):
        self.qNetwork1.setTargetNetworks(self.qNetwork1Target)
        self.qNetwork1.setPolicyNetwork(self.policyNetwork)
        (
            qNetworkTraining,
            qLoss,
            qRegTerm,
            qBatchwiseLoss,
            nextLogProb,
            nextQuantileValues,
            entropyBonus,
            targetValues,
            targets,
            predictionValues,
            predictions,
            absDiff,
            minorError,
            majorError,
            totalError,
            quantilePunishment,
            quantileRegressionLoss,
            perQuantileLoss
        ) = self.qNetwork1.buildTrainingOperation()

        self.policyNetwork.setQNetwork(self.qNetwork1)
        (
            policyTrainingOperation,
            entropyCoefficientTrainingOperation,
            policyRegTerm,
            entropyCoefficientLoss,
            entropyCoefficient
        ) = self.policyNetwork.buildTrainingOperation()

        softCopy1 = self.qNetwork1Target.buildSoftCopyOperation(self.qNetwork1, self.tau)

        self.hardCopy1 = self.qNetwork1Target.buildSoftCopyOperation(self.qNetwork1, 1.0)

        with self.graph.as_default():
            self.trainingOperations = [
                qNetworkTraining,
                policyTrainingOperation,
                entropyCoefficientTrainingOperation,
                softCopy1,
                qLoss,
                qBatchwiseLoss,
                qRegTerm,
                policyRegTerm,
                entropyCoefficientLoss,
                entropyCoefficient
            ]
    def createActors(self):
        deterministic=False
        self.actors.append(Actor(
            "thread_a",
            deterministic,
            self.envName,
            self.actionOperations,
            self.actionScaling,
            self.statePh,
            self.memoryBuffer,
            0.0,
            self.numActions,
            self.numStateVariables,
            self.randomStartSteps,
            self.gamma,
            self.rewardScaling,
            self.maxTrainSteps,
            self.maxEpisodes,
            self.sess
        ))
    def execute(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        if self.train:
            self.sess.run([self.hardCopy1])
        for a in self.actors:
            thread = threading.Thread(target=self.executeActor, args=(a,))
            thread.start()
        thread = threading.Thread(target=self.executeTrainer, args=())
        thread.start()
    def executeActor(self, actor):
        actor.execute()
    def executeTrainer(self):
        self.trainer.execute()
            


