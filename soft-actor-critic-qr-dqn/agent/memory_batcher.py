import numpy as np
import random
from collections import deque
from . import constants

class MemoryBatcher:
    def __init__(self,
                 numMemories,
                 batchSize,
                 nStep):
        self.buffer = deque([], numMemories)
        self.batchSize = batchSize
        self.nStep = nStep
        self.minLoss = 1
        self.clearEpisodeBuffer()
    def clearEpisodeBuffer(self):
        self.episodeBuffer = np.empty((0, constants.NUM_MEMORY_ENTRIES))
    def add(self, memory):
        memory[constants.LOSS] = 1
        memory[constants.PRIORITY] = 1
        self.episodeBuffer = np.append(self.episodeBuffer, [memory], axis=0)
    def endEpisode(self):
        for idx in range(len(self.episodeBuffer)):
            finalIdx = min(idx + self.nStep - 1, len(self.episodeBuffer) - 1)
            rows = self.episodeBuffer[idx:finalIdx+1,:]
            rewards = rows[:,constants.REWARD]
            firstRow = rows[0]
            lastRow = rows[-1]
            totalReward = np.sum(rewards)
            newMemoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
            newMemoryEntry[constants.STATE] = firstRow[constants.STATE]
            newMemoryEntry[constants.ACTION] = firstRow[constants.ACTION]
            newMemoryEntry[constants.REWARD] = totalReward
            newMemoryEntry[constants.NEXT_STATE] = lastRow[constants.NEXT_STATE]
            newMemoryEntry[constants.GAMMA] = firstRow[constants.GAMMA]
            newMemoryEntry[constants.IS_TERMINAL] = lastRow[constants.IS_TERMINAL]
            self.buffer.append(newMemoryEntry)
        self.clearEpisodeBuffer()
    def clear(self):
        self.buffer.clear()
    def getMemoryBatch(self):
        return random.sample(self.buffer, min(self.batchSize, len(self.buffer)))
    def updateMemories(self, memories):
        return