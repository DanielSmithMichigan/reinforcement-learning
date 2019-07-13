import numpy as np
import random
from collections import deque
from . import constants

class SimpleExperienceReplay:
    def __init__(self,
                 numMemories,
                 batchSize):
        self.buffer = deque([], numMemories)
        self.batchSize = batchSize
        self.minLoss = 1
    def add(self, memory):
        memory[constants.LOSS] = 1
        memory[constants.PRIORITY] = 1
        self.buffer.append(memory)
    def clear(self):
        self.buffer.clear()
    def getMemoryBatch(self):
        return random.sample(self.buffer, min(self.batchSize, len(self.buffer)))
    def updateMemories(self, memories):
        return