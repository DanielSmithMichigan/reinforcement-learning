from agent.network import Network
import random
import tensorflow as tf
import unittest
from agent import constants
from agent import util
import numpy as np
import sys

# def trainNetworkTo(sess, network, bestAction, numTrainingIterations, maxAvailableActions):
#     for i in range(numTrainingIterations):
#         action = random.randint(0, maxAvailableActions - 1)
#         reward = 1 if action == bestAction else 0
#         memory = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
#         memory[constants.STATE] = [random.random()]
#         memory[constants.ACTION] = action
#         memory[constants.REWARD] = reward
#         memory[constants.NEXT_STATE] = [random.random()]
#         memory[constants.GAMMA] = 0
#         network.trainAgainst([memory, memory])

# def testNetworkForAction(assertEqual, sess, network, bestAction, numTestingIterations):
#     for i in range(numTestingIterations):
#         chosenAction, qValues = sess.run([network.chosenAction, network.qValues], feed_dict={
#             network.actionInput: [0,1,0],
#             network.environmentInput: [
#                 [random.random()],
#                 [random.random()],
#                 [random.random()]
#             ]
#         })
#         for i in range(len(chosenAction)):
#             assertEqual(chosenAction[i], bestAction)

class TestNetwork(unittest.TestCase):
    def testQuantilesShape(self):
        with tf.Session() as sess:
            targetNetwork = Network(
                name="target-network",
                sess=sess,
                numObservations=1,
                numAvailableActions=2,
                preNetworkSize=[6],
                postNetworkSize=[7],
                numQuantiles=8,
                embeddingDimension=9,
                learningRate=1e-5,
                maxGradientNorm=5,
                batchSize=64,
                kappa=1.1
            )
            n = Network(
                name="learning-network",
                sess=sess,
                numObservations=1,
                numAvailableActions=2,
                preNetworkSize=[6],
                postNetworkSize=[7],
                numQuantiles=8,
                embeddingDimension=9,
                learningRate=1e-5,
                maxGradientNorm=5,
                batchSize=64,
                kappa=1.1,
                targetNetwork=targetNetwork
            )
            sess.run(tf.global_variables_initializer())
            (
                quantilesEmbeddingBox,
                cosineTimestep,
                tiledQuantilesPreCos,
                firstJoinedLayer,
                quantileValues,
                indexedQuantiles,
                actionInput,
                quantileDistance,
                observedRewards,
                totalQuantileError,
                belowQuantile,
                sizedQuantiles,
                quantileThresholds,
                qValues,
                chosenAction,
                maxQ,
                targetNetworkIndexedQuantiles,
                tiledTargetValues,
                gammas,
                targetValues,
                predictedValues,
                quantileRegressionLoss
            ) = sess.run([
                n.quantilesEmbeddingBox,
                n.cosineTimestep,
                n.tiledQuantilesPreCos,
                n.firstJoinedLayer,
                n.quantileValues,
                n.indexedQuantiles,
                n.actionInput,
                n.quantileDistance,
                n.observedRewards,
                n.totalQuantileError,
                n.belowQuantile,
                n.sizedQuantiles,
                n.quantileThresholds,
                n.qValues,
                n.chosenAction,
                n.maxQ,
                targetNetwork.indexedQuantiles,
                n.tiledTargetValues,
                n.gammas,
                n.targetValues,
                n.predictedValues,
                n.quantileRegressionLoss
            ], feed_dict={
                n.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(64, 8)),
                n.environmentInput: np.random.uniform(low=0.0, high=1.0, size=(64, 1)),
                n.actionInput: np.random.randint(2, size=(64)),
                n.gammas: np.random.uniform(low=0.0, high=1.0, size=(64)),
                n.observedRewards: np.random.uniform(low=-200.0, high=200.0, size=(64)),
                targetNetwork.environmentInput: np.random.uniform(low=0.0, high=1.0, size=(64, 1)),
                targetNetwork.actionInput: np.random.randint(2, size=(64)),
                targetNetwork.quantileThresholds: np.random.uniform(low=0.0, high=1.0, size=(64, 8))
            })
            np.testing.assert_equal(np.shape(quantilesEmbeddingBox), [64, 8, 9])
            for x in range(64):
                for y in range(8):
                    for z in range(9):
                        if x != 0:
                            self.assertNotEqual(quantilesEmbeddingBox[0][y][z],quantilesEmbeddingBox[x][y][z])
                        if y != 0:
                            self.assertNotEqual(quantilesEmbeddingBox[x][0][z],quantilesEmbeddingBox[x][y][z])
                        self.assertEqual(quantilesEmbeddingBox[x][y][0],quantilesEmbeddingBox[x][y][z])
            np.testing.assert_equal(np.shape(cosineTimestep), [1, 1, 9])
            self.assertEqual(cosineTimestep[0][0][0], 1)
            self.assertEqual(cosineTimestep[0][0][8], 9)
            for x in range(64):
                for y in range(8):
                    for z in range(9):
                        self.assertEqual(tiledQuantilesPreCos[x][y][z], cosineTimestep[0][0][z] * quantilesEmbeddingBox[x][y][0])
            np.testing.assert_equal(np.shape(firstJoinedLayer), [64, 8, 6])
            np.testing.assert_equal(np.shape(quantileValues), [64, 2, 8])
            np.testing.assert_equal(np.shape(indexedQuantiles), [64, 8])
            np.testing.assert_equal(np.shape(predictedValues), [64, 8, 8])
            np.testing.assert_equal(np.shape(targetValues), [64, 8, 8])
            np.testing.assert_equal(np.shape(totalQuantileError), [64, 8, 8])
            np.testing.assert_equal(np.shape(quantileRegressionLoss), [64, 8, 8])
            for batchIndex in range(64):
                for quantileIndex in range(8):
                    self.assertEqual(indexedQuantiles[batchIndex][quantileIndex], quantileValues[batchIndex][actionInput[batchIndex]][quantileIndex])
                    for quantileIndex_2 in range(8):
                        self.assertEqual(predictedValues[batchIndex][quantileIndex][quantileIndex], predictedValues[batchIndex][quantileIndex][quantileIndex_2])
                    for targetNetworkQuantileIndex in range(8):
                        for quantileIndex_2 in range(8):
                            self.assertEqual(tiledTargetValues[batchIndex][quantileIndex][targetNetworkQuantileIndex],tiledTargetValues[batchIndex][quantileIndex_2][targetNetworkQuantileIndex])    
                        self.assertEqual(tiledTargetValues[batchIndex][quantileIndex][targetNetworkQuantileIndex], targetNetworkIndexedQuantiles[batchIndex][targetNetworkQuantileIndex])
                        self.assertAlmostEqual(targetValues[batchIndex][quantileIndex][targetNetworkQuantileIndex], targetNetworkIndexedQuantiles[batchIndex][targetNetworkQuantileIndex] * gammas[batchIndex] + observedRewards[batchIndex], 5)
                        currentQuantileDistance = quantileDistance[batchIndex][quantileIndex][targetNetworkQuantileIndex]
                        self.assertAlmostEqual(currentQuantileDistance, (targetNetworkIndexedQuantiles[batchIndex][targetNetworkQuantileIndex] * gammas[batchIndex] + observedRewards[batchIndex]) - indexedQuantiles[batchIndex][quantileIndex], 5)
                        if abs(currentQuantileDistance) <= 1.1:
                            self.assertAlmostEqual(totalQuantileError[batchIndex][quantileIndex][targetNetworkQuantileIndex], 0.5 * currentQuantileDistance ** 2, 4)
                        else:
                            self.assertAlmostEqual(totalQuantileError[batchIndex][quantileIndex][targetNetworkQuantileIndex], 1.1 * (abs(currentQuantileDistance) - 0.5 * 1.1), 4)
                        self.assertEqual(bool(belowQuantile[batchIndex][quantileIndex][targetNetworkQuantileIndex]), currentQuantileDistance < 0)
                        if currentQuantileDistance > 0:
                            self.assertAlmostEqual(sizedQuantiles[batchIndex][quantileIndex][targetNetworkQuantileIndex], quantileThresholds[batchIndex][quantileIndex], 7)
                        else:
                            self.assertAlmostEqual(sizedQuantiles[batchIndex][quantileIndex][targetNetworkQuantileIndex], 1 - quantileThresholds[batchIndex][quantileIndex], 7)
            # for batchIndex in range(64):
            #     for action in range(2):
            #         self.assertAlmostEqual(qValues[batchIndex][action], np.mean(quantileValues[batchIndex][action]), 5)
            #         self.assertAlmostEqual(maxQ[batchIndex], np.max(qValues[batchIndex]), 5)
            #         self.assertEqual(chosenAction[batchIndex], np.argmax(qValues[batchIndex]))

    # def testQValues(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-2",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=1,
    #             learningRate=1e-5,
    #             numAtoms=51,
    #             valueMin=-200.0,
    #             valueMax=200.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         probabilities = sess.run(n.probabilities, feed_dict={
    #             n.environmentInput: [[0.0]]
    #         })
    #         self.assertEqual(np.shape(probabilities), (1, 1, 51))
    # def testActionProbabilities(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-3",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=4,
    #             valueMin=-200.0,
    #             valueMax=200.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         actionLogits, logits = sess.run([n.actionLogits, n.logits], feed_dict={
    #             n.environmentInput: [[0.0], [0.2], [-0.1]],
    #             n.actionInput: [2, 0, 1]
    #         })
    #         self.assertEqual(np.shape(actionLogits), (3, 4))
    #         np.testing.assert_equal(actionLogits[0], logits[0][2])
    #         np.testing.assert_equal(actionLogits[1], logits[1][0])
    #         np.testing.assert_equal(actionLogits[2], logits[2][1])
    # def testQValues(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-4",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         qValues, probabilities, support = sess.run([n.qValues, n.probabilities, n.support], feed_dict={
    #             n.environmentInput: [[0.0], [1.0], [0.5]]
    #         })
    #         self.assertEqual(np.shape(qValues), (3, 3))
    #         for _batchIndex in range(3):
    #             for _actionIndex in range(3):
    #                 probabilityDistribution = probabilities[_batchIndex][_actionIndex]
    #                 self.assertEqual(np.shape(probabilityDistribution), (3,))
    #                 distributionTotal = 0.0
    #                 for _supportIndex in range(3):
    #                     distributionTotal = distributionTotal + probabilityDistribution[_supportIndex] * support[_supportIndex]
    #                 self.assertAlmostEqual(distributionTotal, qValues[_batchIndex][_actionIndex] + 0.0, 5)
    # def testMaxQ(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-5",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         qValues, chosenAction, maxQ = sess.run([n.qValues, n.chosenAction, n.maxQ], feed_dict={
    #             n.environmentInput: [[0.0], [1.0], [0.5]]
    #         })
    #         for _batchIndex in range(3):
    #             batchQs = qValues[_batchIndex]
    #             maxBatchQ = np.max(batchQs)
    #             maxBatchQIndex = np.argmax(batchQs)
    #             self.assertEqual(maxBatchQ, maxQ[_batchIndex])
    #             self.assertEqual(maxBatchQIndex, chosenAction[_batchIndex])
    # def testChosenActionProbabilities(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-6",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         probabilities, chosenAction, chosenActionProbabilities = sess.run([n.probabilities, n.chosenAction, n.chosenActionProbabilities], feed_dict={
    #             n.environmentInput: [[0.0], [1.0], [0.5]]
    #         })
    #         for _batchIndex in range(3):
    #             np.testing.assert_equal(probabilities[_batchIndex][chosenAction[_batchIndex]], chosenActionProbabilities[_batchIndex])
    # def testTrainForDistribution(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-7",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[64,64,64],
    #             numAvailableActions=1,
    #             learningRate=1e-3,
    #             numAtoms=51,
    #             valueMin=-10.0,
    #             valueMax=10.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         dist = np.zeros(51)
    #         dist[0] = 1
    #         for i in range(5000):
    #             _, loss = sess.run([n.trainingOperation, n.loss], feed_dict={
    #                 n.actionInput: [0],
    #                 n.targetProbabilities: [dist],
    #                 n.environmentInput: [
    #                     [np.random.random()]
    #                 ]
    #             })
    #         loss = sess.run([n.loss], feed_dict={
    #             n.actionInput: [0],
    #             n.targetProbabilities: [dist],
    #             n.environmentInput: [
    #                 [np.random.random()]
    #             ]
    #         })
    #         self.assertEqual(loss[0] < 1e-3, True)
    # def testTrainTwoNetworks(self):
    #     with tf.Session() as sess:
    #         learnedNetwork = Network(
    #             name="network-8",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[64,64,64],
    #             numAvailableActions=1,
    #             learningRate=1e-3,
    #             numAtoms=51,
    #             valueMin=-10.0,
    #             valueMax=10.0
    #         )
    #         targetNetwork = Network(
    #             name="network-9",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[64,64,64],
    #             numAvailableActions=1,
    #             learningRate=1e-3,
    #             numAtoms=51,
    #             valueMin=-10.0,
    #             valueMax=10.0
    #         )
    #         targetNetwork.makeDuplicationOperation(learnedNetwork.networkParams)
    #         learnedNetwork.makeDuplicationOperation(targetNetwork.networkParams)
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(learnedNetwork.duplicateOtherNetwork)
    #         batchSize = 32
    #         trainingPerTargetUpdate = 256
    #         targetUpdates = 12
    #         for i in range(targetUpdates):
    #             for j in range(trainingPerTargetUpdate):
    #                 batch = []
    #                 for k in range(batchSize):
    #                     memory = np.zeros(constants.NUM_MEMORY_ENTRIES, dtype=object)
    #                     memory[constants.ACTION] = 0
    #                     memory[constants.STATE] = [np.random.random()]
    #                     memory[constants.REWARD] = 1
    #                     memory[constants.NEXT_STATE] = [np.random.random()]
    #                     memory[constants.GAMMA] = 0
    #                     memory[constants.IS_TERMINAL] = True
    #                     batch.append(memory)
    #                 targetProbabilities = targetNetwork.getTargetDistributions(batch)
    #                 _, loss = sess.run([learnedNetwork.trainingOperation, learnedNetwork.loss], feed_dict={
    #                     learnedNetwork.actionInput: util.getColumn(batch, constants.ACTION),
    #                     learnedNetwork.targetProbabilities: targetProbabilities,
    #                     learnedNetwork.environmentInput: util.getColumn(batch, constants.STATE)
    #                 })
    #                 print("LOSS: ",np.mean(loss))
    #             sess.run(targetNetwork.duplicateOtherNetwork)
    #             print("TARGET UPDATED")
            # self.assertEqual(loss[0] < 1e-3, True)
    # def testTrainIt(self):
    #     NUM_TRAINING_ITERATIONS = 100
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-7",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16,16],
    #             numAvailableActions=5,
    #             learningRate=1e-3,
    #             numAtoms=4,
    #             valueMin=0.0,
    #             valueMax=10
    #         )
    #         for testEpisode in range(2):
    #             for bestAction in range(5):
    #                 sess.run(tf.global_variables_initializer())
    #                 trainNetworkTo(sess, n, bestAction, NUM_TRAINING_ITERATIONS, 5)
    #                 testNetworkForAction(self.assertEqual, sess, n, bestAction, NUM_TRAINING_ITERATIONS)
#     def testAssignment(self):
#         NUM_TRAINING_ITERATIONS = 100
#         with tf.Session() as sess:
#             n = Network(
#                 name="network-a",
#                 sess=sess,
#                 numObservations=1,
#                 networkSize=[16,16],
#                 numAvailableActions=5,
#                 learningRate=1e-3,
#                 numAtoms=4,
#                 valueMin=0.0,
#                 valueMax=10
#             )
#             n2 = Network(
#                 name="network-b",
#                 sess=sess,
#                 numObservations=1,
#                 networkSize=[16,16],
#                 numAvailableActions=5,
#                 learningRate=1e-3,
#                 numAtoms=4,
#                 valueMin=0.0,
#                 valueMax=10
#             )
#             n.makeDuplicationOperation(n2.networkParams)
#             sess.run(tf.global_variables_initializer())
#             trainNetworkTo(sess, n, 1, NUM_TRAINING_ITERATIONS, 5)
#             testNetworkForAction(self.assertEqual, sess, n, 1, NUM_TRAINING_ITERATIONS)
#             trainNetworkTo(sess, n2, 2, NUM_TRAINING_ITERATIONS, 5)
#             testNetworkForAction(self.assertEqual, sess, n2, 2, NUM_TRAINING_ITERATIONS)
#             sess.run(n.duplicateOtherNetwork)
#             testNetworkForAction(self.assertEqual, sess, n, 2, NUM_TRAINING_ITERATIONS)
#             testNetworkForAction(self.assertEqual, sess, n2, 2, NUM_TRAINING_ITERATIONS)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
