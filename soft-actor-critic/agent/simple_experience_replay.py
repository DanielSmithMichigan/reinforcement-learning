import numpy as np
import random
from collections import deque

class SimpleExperienceReplay:
    def __init__(self,
                 numMemories,
                 batchSize):
        self.buffer = deque([], numMemories)
        self.batchSize = batchSize
    def add(self, memory):
        self.buffer.append(memory)
    def getMemoryBatch(self):
        return random.sample(self.buffer, min(self.batchSize, len(self.buffer)))