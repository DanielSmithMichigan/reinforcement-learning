# import tensorflow as tf
# import numpy as np
# import unittest
# import math
# from agent.PolicyNetwork import PolicyNetwork
# from agent import constants
# from agent import util
# import matplotlib
# import matplotlib.pyplot as plt
# import time
# plt.ion()
# #     memories = []
# #     for j in range(64):
# #         memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
# #         memoryEntry[constants.STATE] = [np.random.random()]
# #         memoryEntry[constants.ACTION] = [np.random.random()]
# #         memoryEntry[constants.REWARD] = [np.random.random()]
# #         memoryEntry[constants.NEXT_STATE] = [np.random.random()]
# #         memoryEntry[constants.GAMMA] = 0
# #         memoryEntry[constants.IS_TERMINAL] = False
# #         memories.append(memoryEntry)
# #     policyNetwork.trainAgainst(memories)
# #     action = policyNetwork.getAction([np.random.random()])
# #     qValue = sess.run(q.qValue, feed_dict={
# #         q.actionsPh: [action],
# #         q.statePh: np.reshape([np.random.random()], [-1, 1])
# #     })
# #     print("action: ",action[0]," qValue: ",qValue[0][0])

# def getRandomObservations(batchSize, numSamples):
#     output = []
#     for i in range(batchSize):
#         batch = []
#         for j in range(numSamples):
#             batch.append(np.random.random())
#         output.append(batch)
#     return output


# class TestPolicyNetwork(unittest.TestCase):
#     def testShape(self):
#         sess = tf.Session()
#         numActions = 7
#         batchSize = 36
#         numObservations = 12
#         policyNetwork = PolicyNetwork(
#             sess=sess,
#             name="PolicyNetwork_"+str(np.random.randint(1000000,9999999)),
#             numStateVariables=numObservations,
#             numActions=numActions,
#             networkSize=[256,256],
#             entropyCoefficient=1.0,
#             learningRate=2e-4,
#             maxGradientNorm=5.0,
#             batchSize=64,
#             weightRegularizationConstant=0.1
#         )
#         q = type('', (), {})()
#         q.actionsPh = tf.identity(policyNetwork.actionsChosen, name="Actions")
#         q.statePh = tf.placeholder(tf.float32, [None, numObservations], name="State")
#         q.qValue = tf.reduce_mean(-tf.abs(.25 - q.actionsPh), axis=1)
#         policyNetwork.setQNetwork(q)
#         policyNetwork.buildTrainingOperation()
#         sess.run(tf.global_variables_initializer())
#         (
#             variance,
#             mean,
#             rawAction,
#             actionsChosen
#         ) = sess.run([
#             policyNetwork.logScaleActionVariance,
#             policyNetwork.actionMean,
#             policyNetwork.rawAction,
#             policyNetwork.actionsChosen
#         ], feed_dict={
#             policyNetwork.statePh: getRandomObservations(batchSize, numObservations)
#         })
#         np.testing.assert_array_equal(np.shape(variance), [batchSize, numActions])
#         np.testing.assert_array_equal(np.shape(mean), [batchSize, numActions])
#         np.testing.assert_array_equal(np.shape(rawAction), [batchSize, numActions])
#         np.testing.assert_array_equal(np.shape(actionsChosen), [batchSize, numActions])
#     def testDistributions(self):
#         sess = tf.Session()
#         numActions = 1
#         batchSize = 200000
#         numObservations = 12
#         policyNetwork = PolicyNetwork(
#             sess=sess,
#             name="PolicyNetwork_"+str(np.random.randint(1000000,9999999)),
#             numStateVariables=numObservations,
#             numActions=numActions,
#             networkSize=[256,256],
#             entropyCoefficient=1.0,
#             learningRate=2e-4,
#             maxGradientNorm=5.0,
#             batchSize=64,
#             weightRegularizationConstant=0.1
#         )
#         q = type('', (), {})()
#         q.actionsPh = tf.identity(policyNetwork.actionsChosen, name="Actions")
#         q.statePh = tf.placeholder(tf.float32, [None, numObservations], name="State")
#         q.qValue = tf.reduce_mean(-tf.abs(.25 - q.actionsPh), axis=1)
#         policyNetwork.setQNetwork(q)
#         policyNetwork.buildTrainingOperation()
#         sess.run(tf.global_variables_initializer())
#         (
#             variance,
#             mean,
#             rawAction
#         ) = sess.run([
#             policyNetwork.logScaleActionVariance,
#             policyNetwork.actionMean,
#             policyNetwork.rawAction
#         ], feed_dict={
#             policyNetwork.actionMean: np.tile(2, (batchSize, numActions)),
#             policyNetwork.logScaleActionVariance: np.tile(math.log(5), (batchSize, numActions)),
#             policyNetwork.statePh: getRandomObservations(batchSize, numObservations)
#         })
#         np.testing.assert_array_equal(np.shape(variance), [batchSize, numActions])
#         np.testing.assert_array_equal(np.shape(mean), [batchSize, numActions])
#         np.testing.assert_almost_equal(np.mean(rawAction),2,decimal=1)
#         np.testing.assert_almost_equal(np.std(rawAction),5,decimal=1)
#     def testQLearning(self):
#         for targetBase in range(-40, 40, 5):
#             target = float(targetBase / 100)
#             sess = tf.Session()
#             numActions = 1
#             batchSize = 1
#             numObservations = 1
#             policyNetwork = PolicyNetwork(
#                 sess=sess,
#                 name="PolicyNetwork_"+str(np.random.randint(1000000,9999999)),
#                 numStateVariables=numObservations,
#                 numActions=numActions,
#                 networkSize=[256,256],
#                 entropyCoefficient=0.0,
#                 learningRate=4e-3,
#                 maxGradientNorm=5.0,
#                 batchSize=64,
#                 weightRegularizationConstant=0.05
#             )
#             q = type('', (), {})()
#             q.actionsPh = policyNetwork.actionsChosen
#             q.statePh = tf.placeholder(tf.float32, [None, numObservations], name="State")
#             q.qValue = tf.reduce_mean(-tf.abs(target - q.actionsPh), axis=1)
#             policyNetwork.setQNetwork(q)
#             policyNetwork.buildTrainingOperation()
#             sess.run(tf.global_variables_initializer())
#             for i in range(10000):
#                 memories = []
#                 for j in range(4):
#                     memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
#                     memoryEntry[constants.STATE] = [np.random.random()]
#                     memoryEntry[constants.ACTION] = [np.random.random()]
#                     memoryEntry[constants.REWARD] = [np.random.random()]
#                     memoryEntry[constants.NEXT_STATE] = [np.random.random()]
#                     memoryEntry[constants.GAMMA] = 0
#                     memoryEntry[constants.IS_TERMINAL] = False
#                     memories.append(memoryEntry)
#                 policyNetwork.trainAgainst(memories)
#             allActions = []
#             for i in range(100):
#                 allActions.append(policyNetwork.getAction([np.random.random()])[0])
#             np.testing.assert_almost_equal(np.mean(allActions),target,decimal=2)
#     def testEntropyMaximization(self):
#         sess = tf.Session()
#         numActions = 1
#         batchSize = 1
#         numObservations = 1
#         entropyCoefficient = 0.01
#         policyNetwork = PolicyNetwork(
#             sess=sess,
#             name="PolicyNetwork_"+str(np.random.randint(1000000,9999999)),
#             numStateVariables=numObservations,
#             numActions=numActions,
#             networkSize=[256,256],
#             entropyCoefficient=entropyCoefficient,
#             learningRate=4e-3,
#             maxGradientNorm=5.0,
#             batchSize=batchSize,
#             weightRegularizationConstant=0.1
#         )
#         q = type('', (), {})()
#         q.actionsPh = policyNetwork.actionsChosen
#         q.statePh = tf.placeholder(tf.float32, [None, numObservations], name="State")
#         q.qValue = tf.constant(0.01)
#         policyNetwork.setQNetwork(q)
#         policyNetwork.buildTrainingOperation()
#         sess.run(tf.global_variables_initializer())
#         allStd = []
#         allLog = []
#         allLogProb = []
#         allEntropy = []
#         allMean = []
#         def getStdFor(entropy):
#             policyNetwork.entropyCoefficient = entropy
#             std = None
#             for i in range(10):
#                 memories = []
#                 for j in range(5):
#                     memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
#                     memoryEntry[constants.STATE] = [np.random.random()]
#                     memoryEntry[constants.ACTION] = [np.random.random()]
#                     memoryEntry[constants.REWARD] = [np.random.random()]
#                     memoryEntry[constants.NEXT_STATE] = [np.random.random()]
#                     memoryEntry[constants.GAMMA] = 0
#                     memoryEntry[constants.IS_TERMINAL] = False
#                     memories.append(memoryEntry)
#                 policyNetwork.trainAgainst(memories)
#                 allActions = []
#                 logs = []
#                 logProbs = []
#                 entropys = []
#                 for k in range(1000):
#                     action, log, logProb, entropy = policyNetwork.getAction([np.random.random()])
#                     allActions.append(action)
#                     logs.append(log)
#                     logProbs.append(logProb)
#                     entropys.append(entropy)
#                 std = np.std(allActions) 
#                 allLog.append(np.mean(logs))
#                 allLogProb.append(np.mean(logProbs))
#                 allEntropy.append(np.mean(entropys))
#                 allStd.append(std)
#                 allMean.append(np.mean(allActions))
#             return std
#         positiveStdOne = getStdFor(entropyCoefficient)
#         negativeStdOne = getStdFor(-entropyCoefficient)
#         # print("Positive 1: "+str(positiveStdOne))
#         # print("Negative 1: "+str(negativeStdOne))
#         assert(positiveStdOne > negativeStdOne)