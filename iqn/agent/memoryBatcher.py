import numpy as np
import math
from . import constants
from . import util

class MemoryBatcher:
    def __init__(self,
        nStepReturns):
        self.nStepReturns = nStepReturns
    def batch(self, memories):
        newMemories = []
        memories.reverse()
        while len(memories) >= self.nStepReturns:
            newMemory = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
            newMemory[constants.STATE] = memories[0][constants.STATE]
            newMemory[constants.ACTION] = memories[0][constants.ACTION]
            newMemory[constants.REWARD] = np.sum(util.getColumn(memories, constants.REWARD)[0:self.nStepReturns])
            newMemory[constants.NEXT_STATE] = memories[self.nStepReturns - 1][constants.NEXT_STATE]
            newMemory[constants.GAMMA] = memories[self.nStepReturns - 1][constants.GAMMA]
            newMemory[constants.IS_TERMINAL] = memories[self.nStepReturns - 1][constants.IS_TERMINAL]
            newMemories.append(newMemory)
            for i in range(self.nStepReturns):
                memories.pop(0)
        newMemories.reverse()
        return newMemories








        