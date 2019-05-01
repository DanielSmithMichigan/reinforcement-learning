import numpy as np
import math
from . import constants

class PrioritizedExperienceReplay:
    def __init__(self,
                 numMemories,
                 priorityExponent,
                 batchSize):
        self.maxLoss = 1
        self.numTreeNodes = numMemories - 1
        self.numLeafNodes = numMemories
        self.length = 0
        self.firstLeafPosition = self.numTreeNodes
        self.internalPosition = self.firstLeafPosition
        self.memoryLength = self.numTreeNodes + self.numLeafNodes
        self.nodes = np.array(np.zeros((self.memoryLength, constants.NUM_MEMORY_ENTRIES)), dtype=object)
        self.priorityExponent = priorityExponent
        self.batchSize = batchSize
    def add(self, memory):
        position = self.incrementInternalPosition()
        oldMemory = self.nodes[position]
        memory[constants.POSITION] = position
        memory[constants.LOSS] = max(memory[constants.LOSS], self.maxLoss)
        memory[constants.OLD_PRIORITY] = oldMemory[constants.PRIORITY]
        self.nodes[position] = memory
        self.updateLeaf(memory[constants.POSITION])
        self.length = min(self.length + 1, self.numLeafNodes)
    def incrementInternalPosition(self):
        output = self.internalPosition
        self.internalPosition = self.internalPosition + 1
        if self.internalPosition >= self.memoryLength:
            self.internalPosition = self.firstLeafPosition
        return int(output)
    def updateMemories(self, memories):
        for memory in memories:
            self.nodes[int(memory[constants.POSITION])][constants.LOSS] = memory[constants.LOSS]
            self.updateLeaf(memory[constants.POSITION])
    def updateLeaf(self, position):
        memory = self.nodes[int(position)]
        self.maxLoss = max(self.maxLoss, memory[constants.LOSS])
        memory[constants.PRIORITY] = memory[constants.LOSS] ** self.priorityExponent
        delta = memory[constants.PRIORITY] - memory[constants.OLD_PRIORITY]
        self.updateNextTreeNode(delta, memory[constants.POSITION])
        memory[constants.OLD_PRIORITY] = memory[constants.PRIORITY]
    def updateNextTreeNode(self, delta, position):
        nextTreeNode = math.floor((position - 1) / 2)
        if nextTreeNode >= 0:
            self.nodes[nextTreeNode][constants.PRIORITY] = self.nodes[nextTreeNode][constants.PRIORITY] + delta
        if nextTreeNode > 0:
            self.updateNextTreeNode(delta, nextTreeNode)
    def getMemoryBatch(self):
        return [self.getMemory() for i in range(self.batchSize)]
    def getMemory(self):
        totalPriority = self.nodes[0][constants.PRIORITY]
        choice = np.random.random() * totalPriority
        return self.findChosenMemory(choice, 0)
    def findChosenMemory(self, choice, currentPosition):
        if currentPosition >= self.firstLeafPosition:
            return self.nodes[currentPosition]
        nodeLeftIndex = currentPosition * 2 + 1
        nodeLeftLoss = self.nodes[nodeLeftIndex][constants.PRIORITY]
        nodeRightIndex = nodeLeftIndex + 1
        if choice <= nodeLeftLoss:
            return self.findChosenMemory(choice, nodeLeftIndex)
        else:
            return self.findChosenMemory(choice - nodeLeftLoss, nodeRightIndex)





        