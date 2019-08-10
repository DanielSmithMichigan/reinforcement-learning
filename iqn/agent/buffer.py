from collections import deque
from .network import Network
import tensorflow as tf
import numpy as np
from . import constants
import random
from . import util
import math

#TODO: Update learned network periodically + tests
#TODO: Epsilon + tests
#TODO: Tests on execution function
#TODO: Re-read DQN paper

MAX_ITER = 20

class Buffer:
    def __init__(
            self,
            maxMemoryLength,
            nStepUpdate,
            priorityExponent,
            batchSize,
            gamma,
            includeIntermediatePairs,
            sess
        ):
        self.memory = deque([], maxMemoryLength)
        self.nStepUpdate = nStepUpdate
        self.priorityExponent = priorityExponent
        self.gamma = gamma
        self.batchSize = batchSize
        self.includeIntermediatePairs = includeIntermediatePairs
        self.sess = sess
        self.createUpdatePrioritiesOp()
    def addMemory(self, memory):
        self.memory.append(memory)
    def updatePriorities(self):
        newPriorities = self.sess.run(self.newPriority, feed_dict={
            self.lossInputs: util.getColumn(self.memory, constants.PRIORITY_CACHE)    
        })
        for i in range(len(self.memory)):
            self.memory[i][constants.INDEX] = i
            self.memory[i][constants.PRIORITY] = newPriorities[i]
    def updateLossPriorityCache(self, memoryUnits):
        for memoryUnit in memoryUnits:
            if (math.isnan(memoryUnit[constants.LOSS])):
                print("AAA: ",memoryUnit[constants.LOSS])
            memoryUnit[constants.PRIORITY_CACHE] = memoryUnit[constants.LOSS] ** self.priorityExponent 
    def createUpdatePrioritiesOp(self):
        self.lossInputs = tf.placeholder(tf.float64, [None, ])
        self.total = tf.reduce_sum(self.lossInputs)
        self.newPriority = self.lossInputs / self.total
    def getSampleBatch(self):
        self.updatePriorities()
        samples = []
        i = 0
        while(len(samples) < self.batchSize):
            i = i + 1
            validMinorBatch = self.getValidMinorBatches(samples)
            samples = samples + validMinorBatch
            if i > MAX_ITER:
                # print("ITERATION_EXCEEDED")
                break
        return samples
    def getValidMinorBatches(self, samples):
        minorBatches = self.getMinorBatches()
        minorBatchesOut = []
        for minorBatch in minorBatches:
            if self.overlapsTerminal(minorBatch):
                continue
            pairedMinorBatch = self.toPairs(minorBatch)
            if not self.includeIntermediatePairs:
                pairedMinorBatch = [pairedMinorBatch[0]]
            minorBatchesOut = minorBatchesOut + pairedMinorBatch
        return minorBatchesOut
    def toPairs(self, minorBatch):
        paired = []
        lastEntry = minorBatch[-1]
        for i in range(len(minorBatch)):
            newPair = np.zeros_like(lastEntry)
            for j in range(len(minorBatch[i])):
                newPair[j] = minorBatch[i][j]
            newPair[constants.NEXT_STATE] = lastEntry[constants.NEXT_STATE]
            newPair[constants.IS_TERMINAL] = lastEntry[constants.IS_TERMINAL]
            newPair[constants.REWARD] = 0
            cumulativeGamma = 1
            for j in range(i, len(minorBatch)):
                newPair[constants.REWARD] = newPair[constants.REWARD] + cumulativeGamma * minorBatch[j][constants.REWARD]
                cumulativeGamma = cumulativeGamma * self.gamma
            paired.append(newPair)
        return paired
    def overlapsTerminal(self, minorBatch):
        for i in range(len(minorBatch) - 1):
            if (minorBatch[i][constants.IS_TERMINAL]):
                return True
        return False
    def overlapsExistingSample(self, minorBatch, samples):
        indexesA = util.getColumn(minorBatch, constants.INDEX)
        indexesB = util.getColumn(samples, constants.INDEX)
        intersection = np.intersect1d(indexesA, indexesB)
        overlaps = len(intersection) > 0
        return overlaps
    def getMinorBatches(self):
        indexes = np.random.choice(range(len(self.memory)), size=self.batchSize, p=util.getColumn(self.memory, constants.PRIORITY))
        batched = [self.indexToBatch(i) for i in indexes]
        return batched
    def indexToBatch(self, index):
        return [self.memory[i] for i in range(max(index - (self.nStepUpdate - 1), 0), index + 1)]


