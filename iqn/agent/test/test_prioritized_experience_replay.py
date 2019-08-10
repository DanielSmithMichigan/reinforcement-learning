from agent.prioritized_experience_replay import PrioritizedExperienceReplay
from agent import constants
import random
import unittest
import numpy as np
import math

# class TestPrioritizedExperienceReplay(unittest.TestCase):
    # def testOneNode(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=1,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 1)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 0)
    # def testAddingNodeTwice(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=1,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 0)
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 0)
    # def testUpdatingOldPriority(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=1,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 0)
    #     memory[constants.LOSS] = 2
    #     buffer.add(memory)
    #     print(buffer.nodes[0][constants.PRIORITY])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 2)
    # def testUpdatingOldPriorityUsingUpdateMethod(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=1,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 0)
    #     memory[constants.LOSS] = 2
    #     buffer.updateMemories([memory])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 2)
    # def testTwoNodes(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=2,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 3)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 1)
    #     memory[constants.LOSS] = 2
    #     buffer.updateMemories([memory])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 2)
    # def testTwoNodesTwoMemories(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=2,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 3)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 1)
    #     secondMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     secondMemory[constants.LOSS] = 2
    #     buffer.add(secondMemory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 3)
    # def testTwoNodesTwoMemoriesThenUpdate(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=2,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 3)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 1)
    #     secondMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     secondMemory[constants.LOSS] = 2
    #     buffer.add(secondMemory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 3)
    #     memory[constants.LOSS] = 2
    #     buffer.updateMemories([memory])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 4)
    # def testRepeatedUpdates(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=2,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 3)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES, dtype=object)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 1)
    #     secondMemory = np.zeros(constants.NUM_MEMORY_ENTRIES, dtype=object)
    #     secondMemory[constants.LOSS] = 2
    #     buffer.add(secondMemory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 3)
    #     memory[constants.LOSS] = 2
    #     buffer.updateMemories([memory, secondMemory])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 4)
    # def testRepeatedUpdatesTwo(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=2,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 3)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 1)
    #     secondMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     secondMemory[constants.LOSS] = 2
    #     buffer.add(secondMemory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 3)
    #     memory[constants.LOSS] = 2
    #     secondMemory[constants.LOSS] = 4
    #     buffer.updateMemories([memory, secondMemory])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 6)
    # def testSevenNodes(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=4,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 7)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 1)
    #     secondMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     secondMemory[constants.LOSS] = 2
    #     buffer.add(secondMemory)
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 3)
    #     memory[constants.LOSS] = 2
    #     secondMemory[constants.LOSS] = 4
    #     buffer.updateMemories([memory, secondMemory])
    #     self.assertEqual(buffer.nodes[0][constants.PRIORITY], 6)
    # def testGetOneNode(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=4,
    #          priorityExponent=0,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 7)
    #     retrievedMemory = buffer.getMemoryBatch()[0]
    #     self.assertEqual(retrievedMemory[constants.LOSS], 0)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     retrievedMemory = buffer.getMemoryBatch()[0]
    #     self.assertEqual(retrievedMemory[constants.LOSS], 1)
    # def testNeverGetEmpty(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=4,
    #          priorityExponent=0,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 7)
    #     retrievedMemory = buffer.getMemoryBatch()[0]
    #     self.assertEqual(retrievedMemory[constants.LOSS], 0)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 1
    #     buffer.add(memory)
    #     for i in range(10):
    #         retrievedMemory = buffer.getMemoryBatch()[0]
    #         self.assertEqual(retrievedMemory[constants.LOSS], 1)
    # def testTwoMemories(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=4,
    #          priorityExponent=0,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 7)
    #     retrievedMemory = buffer.getMemoryBatch()[0]
    #     self.assertEqual(retrievedMemory[constants.LOSS], 0)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 0
    #     buffer.add(memory)
    #     anotherMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     anotherMemory[constants.LOSS] = 1
    #     buffer.add(anotherMemory)
    #     retrievalCount = np.zeros(2)
    #     for i in range(100):
    #         retrievedMemory = buffer.getMemoryBatch()[0]
    #         retrievalCount[int(retrievedMemory[constants.LOSS])] = retrievalCount[int(retrievedMemory[constants.LOSS])] + 1
    #     self.assertTrue(retrievalCount[0] > 0)
    #     self.assertTrue(retrievalCount[1] > 0)
    # def testTwoMemoriesWithPriority(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=4,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 7)
    #     retrievedMemory = buffer.getMemoryBatch()[0]
    #     self.assertEqual(retrievedMemory[constants.LOSS], 0)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = 0
    #     buffer.add(memory)
    #     anotherMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     anotherMemory[constants.LOSS] = 1
    #     buffer.add(anotherMemory)
    #     retrievalCount = np.zeros(2)
    #     for i in range(100):
    #         retrievedMemory = buffer.getMemoryBatch()[0]
    #         retrievalCount[int(retrievedMemory[constants.LOSS])] = retrievalCount[int(retrievedMemory[constants.LOSS])] + 1
    #     self.assertTrue(retrievalCount[0] == 0)
    #     self.assertTrue(retrievalCount[1] == 100)
    # def testTwoMemoriesWithPriorityTwo(self):
    #     buffer = PrioritizedExperienceReplay(
    #          numMemories=4,
    #          priorityExponent=1,
    #          batchSize=1
    #     )
    #     self.assertEqual(len(buffer.nodes), 7)
    #     retrievedMemory = buffer.getMemoryBatch()[0]
    #     self.assertEqual(retrievedMemory[constants.LOSS], 0)
    #     memory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     memory[constants.LOSS] = .1
    #     buffer.add(memory)
    #     anotherMemory = np.zeros(constants.NUM_MEMORY_ENTRIES)
    #     anotherMemory[constants.LOSS] = 1
    #     buffer.add(anotherMemory)
    #     retrievalCount = np.zeros(2)
    #     for i in range(100):
    #         retrievedMemory = buffer.getMemoryBatch()[0]
    #         retrievalCount[int(math.floor(retrievedMemory[constants.LOSS]))] = retrievalCount[math.floor(int(retrievedMemory[constants.LOSS]))] + 1
    #     self.assertTrue(retrievalCount[0] > 0)
    #     self.assertTrue(retrievalCount[1] > 0)
    #     self.assertTrue(retrievalCount[1] > retrievalCount[0])



