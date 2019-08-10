# from agent.network import Network
# from agent.buffer import Buffer
# from agent import constants
# from agent import util
# import random
# import tensorflow as tf
# import unittest
# import numpy as np
# import gym

# class TestBuffer(unittest.TestCase):
#     def testGetOnlyOne(self):
#         buffer = Buffer(
#             maxMemoryLength=1,
#             nStepUpdate=1,
#             priorityExponent=1,
#             batchSize=1,
#             gamma=1,
#             includeIntermediatePairs=True
#         )
#         memory=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memory[constants.INDEX] = 2
#         memory[constants.LOSS] = 1
#         memory[constants.REWARD] = 1
#         buffer.addMemory(memory)
#         sampleBatch = buffer.getSampleBatch()
#         self.assertEqual(sampleBatch[0][constants.REWARD], 1)
#         self.assertEqual(sampleBatch[0][constants.INDEX], 0)
#         self.assertEqual(len(sampleBatch), 1)
#     def testGetBoth(self):
#         buffer = Buffer(
#             maxMemoryLength=2,
#             nStepUpdate=1,
#             priorityExponent=0,
#             batchSize=128,
#             gamma=1,
#             includeIntermediatePairs=True
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.INDEX] = 1
#         memoryOne[constants.LOSS] = 1
#         memoryOne[constants.REWARD] = 1
#         buffer.addMemory(memoryOne)
#         memoryTwo=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryTwo[constants.INDEX] = 2
#         memoryTwo[constants.LOSS] = 1
#         memoryTwo[constants.REWARD] = 3
#         buffer.addMemory(memoryTwo)
#         sampleBatch = buffer.getSampleBatch()
#         rewards = util.getColumn(sampleBatch, constants.REWARD)
#         self.assertEqual(len(sampleBatch), 2)
#         self.assertTrue(1 in rewards)
#         self.assertTrue(3 in rewards)
#     def testPrioritization(self):
#         buffer = Buffer(
#             maxMemoryLength=2,
#             nStepUpdate=1,
#             priorityExponent=1,
#             batchSize=1,
#             gamma=1,
#             includeIntermediatePairs=True
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.INDEX] = 1
#         memoryOne[constants.LOSS] = 0
#         memoryOne[constants.REWARD] = 1
#         buffer.addMemory(memoryOne)
#         memoryTwo=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryTwo[constants.INDEX] = 2
#         memoryTwo[constants.LOSS] = 1
#         memoryTwo[constants.REWARD] = 3
#         buffer.addMemory(memoryTwo)
#         allBatches = []
#         for i in range(10):
#             sampleBatch = buffer.getSampleBatch()
#             allBatches = allBatches + sampleBatch
#         rewards = util.getColumn(allBatches, constants.REWARD)
#         self.assertEqual(len(allBatches), 10)
#         self.assertTrue(not(1 in rewards))
#         self.assertTrue(3 in rewards)
#     def testNoPrioritization(self):
#         buffer = Buffer(
#             maxMemoryLength=2,
#             nStepUpdate=1,
#             priorityExponent=0,
#             batchSize=1,
#             gamma=1,
#             includeIntermediatePairs=True
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.INDEX] = 1
#         memoryOne[constants.LOSS] = 0
#         memoryOne[constants.REWARD] = 1
#         buffer.addMemory(memoryOne)
#         memoryTwo=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryTwo[constants.INDEX] = 2
#         memoryTwo[constants.LOSS] = 1
#         memoryTwo[constants.REWARD] = 3
#         buffer.addMemory(memoryTwo)
#         allBatches = []
#         for i in range(100):
#             sampleBatch = buffer.getSampleBatch()
#             allBatches = allBatches + sampleBatch
#         rewards = util.getColumn(allBatches, constants.REWARD)
#         self.assertEqual(len(allBatches), 100)
#         self.assertTrue(1 in rewards)
#         self.assertTrue(3 in rewards)
#     def testRetrieveBeyondIndex(self):
#         buffer = Buffer(
#             maxMemoryLength=1,
#             nStepUpdate=3,
#             priorityExponent=0,
#             batchSize=1,
#             gamma=1,
#             includeIntermediatePairs=True
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.INDEX] = 1
#         memoryOne[constants.LOSS] = 0
#         memoryOne[constants.REWARD] = 1
#         buffer.addMemory(memoryOne)
#         sampleBatch = buffer.getSampleBatch()
#         self.assertEqual(len(sampleBatch), 1)
#         self.assertEqual(sampleBatch[0][constants.REWARD], 1)
#     def testHorizon(self):
#         buffer = Buffer(
#             maxMemoryLength=3,
#             nStepUpdate=3,
#             priorityExponent=0,
#             batchSize=3,
#             gamma=.5,
#             includeIntermediatePairs=True
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.INDEX] = 1
#         memoryOne[constants.LOSS] = 0
#         memoryOne[constants.REWARD] = 1
#         memoryOne[constants.STATE] = 7
#         memoryOne[constants.NEXT_STATE] = 2
#         buffer.addMemory(memoryOne)
#         memoryTwo=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryTwo[constants.INDEX] = 1
#         memoryTwo[constants.LOSS] = 0
#         memoryTwo[constants.REWARD] = 2
#         memoryTwo[constants.STATE] = 8
#         memoryTwo[constants.NEXT_STATE] = 3
#         buffer.addMemory(memoryTwo)
#         memoryThree=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryThree[constants.INDEX] = 1
#         memoryThree[constants.LOSS] = 1
#         memoryThree[constants.REWARD] = 3
#         memoryThree[constants.NEXT_STATE] = 4
#         memoryThree[constants.STATE] = 9
#         memoryThree[constants.IS_TERMINAL] = True
#         buffer.addMemory(memoryThree)
#         for testNum in range(10):
#             sampleBatch = buffer.getSampleBatch()
#             lastEntry = sampleBatch[0]
#             for i in range(1, len(sampleBatch)):
#                 currentEntry = sampleBatch[i]
#                 self.assertEqual(currentEntry[constants.STATE], lastEntry[constants.STATE] + 1)
#                 lastEntry = sampleBatch[i]
#     def testTerminal(self):
#         buffer = Buffer(
#             maxMemoryLength=4,
#             nStepUpdate=3,
#             priorityExponent=1,
#             batchSize=20,
#             gamma=.5,
#             includeIntermediatePairs=False
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.LOSS] = 0
#         memoryOne[constants.REWARD] = 1
#         memoryOne[constants.STATE] = 7
#         memoryOne[constants.NEXT_STATE] = 2
#         buffer.addMemory(memoryOne)
#         memoryTwo=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryTwo[constants.LOSS] = 0
#         memoryTwo[constants.REWARD] = 2
#         memoryTwo[constants.STATE] = 8
#         memoryTwo[constants.NEXT_STATE] = 3
#         buffer.addMemory(memoryTwo)
#         memoryThree=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryThree[constants.LOSS] = 1
#         memoryThree[constants.REWARD] = 3
#         memoryThree[constants.IS_TERMINAL] = True
#         memoryThree[constants.STATE] = 9
#         memoryThree[constants.NEXT_STATE] = 4
#         buffer.addMemory(memoryThree)
#         memoryFour=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryFour[constants.LOSS] = 1
#         memoryFour[constants.REWARD] = 3
#         memoryFour[constants.STATE] = 10
#         memoryFour[constants.NEXT_STATE] = 4
#         buffer.addMemory(memoryFour)
#         sampleBatch = buffer.getSampleBatch()
#         self.assertEqual(len(sampleBatch), 1)
#         self.assertEqual(sampleBatch[0][constants.STATE], 7)
#         self.assertEqual(sampleBatch[0][constants.NEXT_STATE], 4)
#         self.assertEqual(sampleBatch[0][constants.REWARD], 2.75)
#     def testTwoValidBatches(self):
#         buffer = Buffer(
#             maxMemoryLength=9,
#             nStepUpdate=3,
#             priorityExponent=0,
#             batchSize=20,
#             gamma=.5,
#             includeIntermediatePairs=False
#         )
#         memoryOne=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryOne[constants.REWARD] = 1
#         memoryOne[constants.STATE] = 10
#         memoryOne[constants.NEXT_STATE] = 1
#         buffer.addMemory(memoryOne)
#         memoryTwo=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryTwo[constants.REWARD] = 2
#         memoryTwo[constants.STATE] = 11
#         memoryTwo[constants.NEXT_STATE] = 2
#         buffer.addMemory(memoryTwo)
#         memoryThree=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryThree[constants.REWARD] = 3
#         memoryThree[constants.STATE] = 12
#         memoryThree[constants.NEXT_STATE] = 3
#         memoryThree[constants.IS_TERMINAL] = True
#         buffer.addMemory(memoryThree)
#         memoryFour=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryFour[constants.REWARD] = 4
#         memoryFour[constants.STATE] = 13
#         memoryFour[constants.NEXT_STATE] = 4
#         buffer.addMemory(memoryFour)
#         memoryFive=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryFive[constants.REWARD] = 5
#         memoryFive[constants.STATE] = 14
#         memoryFive[constants.NEXT_STATE] = 5
#         buffer.addMemory(memoryFive)
#         memorySix=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memorySix[constants.REWARD] = 6
#         memorySix[constants.STATE] = 15
#         memorySix[constants.NEXT_STATE] = 6
#         memorySix[constants.IS_TERMINAL] = True
#         buffer.addMemory(memorySix)
#         memorySeven=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memorySeven[constants.REWARD] = 7
#         memorySeven[constants.STATE] = 16
#         memorySeven[constants.NEXT_STATE] = 7
#         buffer.addMemory(memorySeven)
#         memoryEight=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryEight[constants.REWARD] = 8
#         memoryEight[constants.STATE] = 17
#         memoryEight[constants.NEXT_STATE] = 8
#         buffer.addMemory(memoryEight)
#         memoryNine=np.zeros(constants.NUM_MEMORY_ENTRIES)
#         memoryNine[constants.REWARD] = 9
#         memoryNine[constants.STATE] = 18
#         memoryNine[constants.NEXT_STATE] = 9
#         memoryNine[constants.IS_TERMINAL] = True
#         buffer.addMemory(memoryNine)
#         for i in range(15):
#             sampleBatch = buffer.getSampleBatch()
#             self.assertEqual(len(sampleBatch), 3)
#             newMemoryOne=None
#             newMemoryTwo=None
#             newMemoryThree=None
#             for i in range(len(sampleBatch)):
#                 if sampleBatch[i][constants.STATE] == 10:
#                     newMemoryOne = sampleBatch[i]
#                 if sampleBatch[i][constants.STATE] == 13:
#                     newMemoryTwo = sampleBatch[i]
#                 if sampleBatch[i][constants.STATE] == 16:
#                     newMemoryThree = sampleBatch[i]
#             self.assertEqual(newMemoryOne[constants.STATE], 10)
#             self.assertTrue(newMemoryOne[constants.NEXT_STATE] == 3 or newMemoryOne[constants.NEXT_STATE] == 2 or newMemoryOne[constants.NEXT_STATE] == 1)
#             self.assertTrue(newMemoryOne[constants.REWARD] == 2.75 or newMemoryOne[constants.REWARD] == 2 or newMemoryOne[constants.REWARD] == 1)
#             self.assertEqual(newMemoryTwo[constants.STATE], 13)
#             self.assertEqual(newMemoryTwo[constants.NEXT_STATE], 6)
#             self.assertEqual(newMemoryTwo[constants.REWARD], 8)
#             self.assertEqual(newMemoryThree[constants.STATE], 16)
#             self.assertEqual(newMemoryThree[constants.NEXT_STATE], 9)
#             self.assertEqual(newMemoryThree[constants.REWARD], 13.25)



