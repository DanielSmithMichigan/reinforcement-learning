import tensorflow as tf
import numpy as np
import gym

from . import constants
from . import util

import time

class Trainer:
    def __init__(
        self,
        sess,
        trainingOperations,
        memoryBuffer,
        statePh,
        nextStatePh,
        actionsPh,
        terminalsPh,
        rewardsPh,
        actors,
        minStepsBeforeTraining
    ):
        self.sess = sess
        self.trainingOperations = trainingOperations
        self.memoryBuffer = memoryBuffer
        self.statePh = statePh
        self.nextStatePh = nextStatePh
        self.actionsPh = actionsPh
        self.terminalsPh = terminalsPh
        self.rewardsPh = rewardsPh
        self.actors = actors
        self.minStepsBeforeTraining = minStepsBeforeTraining
        self.trainingSteps = 0
    def trainNetworks(self):
        trainingMemories = self.memoryBuffer.getMemoryBatch()
        (
            qNetwork1Training,
            policyTrainingOperation,
            entropyCoefficientTrainingOperation,
            softCopy1,
            q1Loss,
            q1BatchwiseLoss,
            q1RegTerm,
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
        self.trainingSteps += 1
        for i in range(len(trainingMemories)):
            trainingMemories[i][constants.LOSS] = q1BatchwiseLoss[i]
        self.memoryBuffer.updateMemories(trainingMemories)
    def sumEnvironmentSteps(self):
        total = 0
        for actor in self.actors:
            total += actor.globalStep
        return total
    def execute(self):
        while(True):
            stepsNeeded = self.sumEnvironmentSteps()
            print("NEEDED: "+str(stepsNeeded - self.minStepsBeforeTraining)+" COMPLETED: "+str(self.trainingSteps))
            self.trainUntilCaughtUp()
            time.sleep(1)
    def trainUntilCaughtUp(self):
        environmentSteps = self.sumEnvironmentSteps()
        stepsNeeded = environmentSteps - self.minStepsBeforeTraining
        while(self.trainingSteps < stepsNeeded):
            if(self.trainingSteps % 1000 == 0):
                print("Training steps: "+str(self.trainingSteps)+ " Environment steps: "+str(environmentSteps))
            self.trainNetworks()