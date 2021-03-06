import tensorflow as tf
import numpy as np
import unittest
import math
from agent.QNetwork import QNetwork
from agent.PolicyNetwork import PolicyNetwork
from agent import constants
from agent import util
import matplotlib
import matplotlib.pyplot as plt
import time
plt.ion()

def getRandomObservations(batchSize, numSamples):
    output = []
    for i in range(batchSize):
        batch = []
        for j in range(numSamples):
            batch.append(np.random.random())
        output.append(batch)
    return output


class TestQNetwork(unittest.TestCase):
    def testLossFunction(self):
        graph = tf.Graph()
        numStateVariables = 1
        numActions = 1
        numQuantiles = 8
        embeddingDimension = 7
        with graph.as_default():
            sess = tf.Session()
            statePh = tf.placeholder(tf.float32, [None, numStateVariables], name="State_Placeholder")
            nextStatePh = tf.placeholder(tf.float32, [None, numStateVariables], name="NextState_Placeholder")
            actionsPh = tf.placeholder(tf.float32, [None, numActions], name="Actions_Placeholder")
            rewardsPh = tf.placeholder(tf.float32, [None, ], name="Rewards_Placeholder")
            terminalsPh = tf.placeholder(tf.float32, [None, ], name="Terminals_Placeholder")
            memoryPriorityPh = tf.placeholder(tf.float32, [None, ], name="MemoryPriority_Placeholder")
        batchSize = 4
        gamma = 0.99
        kappa = 1.0
        maxGradientNorm = 5.0 
        targetEntropy = -4.0
        entropyCoefficient = "auto"

        qNetwork = QNetwork(
            sess=sess,
            graph=graph,
            name="QNetwork_1_"+str(np.random.uniform(low=10000,high=99999)),
            numStateVariables=numStateVariables,
            numActions=numActions,
            networkSize=[16,16],
            numQuantiles=numQuantiles,
            gamma=gamma,
            kappa=kappa,
            learningRate=1e-4,
            showGraphs=False,
            statePh=statePh,
            nextStatePh=nextStatePh,
            actionsPh=actionsPh,
            rewardsPh=rewardsPh,
            terminalsPh=terminalsPh,
            memoryPriorityPh=memoryPriorityPh,
            maxGradientNorm=maxGradientNorm
        )

        qNetworkTarget = QNetwork(
            sess=sess,
            graph=graph,
            name="QNetwork_1_Target_"+str(np.random.uniform(low=10000,high=99999)),
            numStateVariables=numStateVariables,
            numActions=numActions,
            networkSize=[16,16],
            numQuantiles=numQuantiles,
            gamma=gamma,
            kappa=kappa,
            learningRate=1e-4,
            showGraphs=False,
            statePh=statePh,
            nextStatePh=nextStatePh,
            actionsPh=actionsPh,
            rewardsPh=rewardsPh,
            terminalsPh=terminalsPh,
            memoryPriorityPh=memoryPriorityPh,
            maxGradientNorm=maxGradientNorm
        )

        policyNetwork = PolicyNetwork(
            sess=sess,
            graph=graph,
            name="PolicyNetwork_"+str(np.random.uniform(low=10000,high=99999)),
            numStateVariables=numStateVariables,
            numActions=numActions,
            networkSize=[16,16],
            learningRate=1e-4,
            batchSize=batchSize,
            showGraphs=False,
            statePh=statePh,
            targetEntropy=targetEntropy,
            entropyCoefficient=entropyCoefficient,
            maxGradientNorm=maxGradientNorm,
            varianceRegularizationConstant=0.00,
            meanRegularizationConstant=0.00
        )
        qNetwork.setTargetNetworks(qNetworkTarget)
        qNetwork.setPolicyNetwork(policyNetwork)
        (
            optimizer,
            loss,
            regTerm,
            batchwiseLoss,
            nextLogProb,
            nextQuantileValues,
            entropyBonus,
            targetValues,
            targets,
            predictionValues,
            predictions,
            absDiff,
            minorError,
            majorError,
            totalError,
            quantilePunishment,
            quantileRegressionLoss,
            perQuantileLoss
        ) = qNetwork.buildTrainingOperation()
        policyNetwork.setQNetwork(qNetwork)
        stateValue = np.random.uniform(size=(batchSize, numStateVariables))
        nextStateValue = np.random.uniform(size=(batchSize, numStateVariables))
        actionsValue = np.random.uniform(size=(batchSize, numActions))
        rewardsValue = np.random.uniform(size=(batchSize,))
        terminalsValue = np.random.randint(low=0,high=2,size=(batchSize,))
        memoryPriorityValue = np.random.uniform(size=(batchSize,))
        quantileThresholdsValue = np.linspace(0, 1, numQuantiles + 2)[1:numQuantiles+1]
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            (
                _,
                lossValues,
                regTermValues,
                batchwiseLossValues,
                nextLogProbValues,
                nextQuantileValuesValues,
                entropyBonusValues,
                targetValuesValues,
                targetsValues,
                predictionValuesValues,
                predictionsValues,
                absDiffValues,
                minorErrorValues,
                majorErrorValues,
                totalErrorValues,
                quantilePunishmentValues,
                quantileRegressionLossValues,
                perQuantileLossValues,
                entropyCoefficientValue
            ) = sess.run([
                optimizer,
                loss,
                regTerm,
                batchwiseLoss,
                nextLogProb,
                nextQuantileValues,
                entropyBonus,
                targetValues,
                targets,
                predictionValues,
                predictions,
                absDiff,
                minorError,
                majorError,
                totalError,
                quantilePunishment,
                quantileRegressionLoss,
                perQuantileLoss,
                policyNetwork.entropyCoefficient
            ], feed_dict={
                statePh: stateValue,
                nextStatePh: nextStateValue,
                actionsPh: actionsValue,
                rewardsPh: rewardsValue,
                terminalsPh: terminalsValue,
                memoryPriorityPh: memoryPriorityValue
            })

            print(np.shape(entropyBonusValues))
            print(np.shape(nextQuantileValuesValues))

            for batchIndex in range(len(targetValuesValues)):
                batch = targetValuesValues[batchIndex]
                for quantileIndex in range(len(batch)):
                    quantileTarget = batch[quantileIndex][0]
                    entropyBonus = -1 * entropyCoefficientValue * nextLogProbValues[batchIndex]
                    nextQValue = nextQuantileValuesValues[batchIndex][quantileIndex]
                    expectedQuantileTarget = rewardsValue[batchIndex] + gamma * (1 - terminalsValue[batchIndex]) * (nextQValue + entropyBonus)
                    np.testing.assert_almost_equal(quantileTarget, expectedQuantileTarget, decimal=4)

            for batchIndex in range(len(targetsValues)):
                for rowIndex in range(len(targetsValues[batchIndex])):
                    for columnIndex in range(len(targetsValues[batchIndex][rowIndex])):
                        np.testing.assert_almost_equal(targetValuesValues[batchIndex][0][columnIndex], targetsValues[batchIndex][rowIndex][columnIndex], decimal=4)
                        np.testing.assert_almost_equal(targetsValues[batchIndex][0][columnIndex], targetsValues[batchIndex][rowIndex][columnIndex], decimal=4)

            for batchIndex in range(len(predictionsValues)):
                for rowIndex in range(len(predictionsValues[batchIndex])):
                    for columnIndex in range(len(predictionsValues[batchIndex][rowIndex])):
                        np.testing.assert_almost_equal(predictionValuesValues[batchIndex][rowIndex][0], predictionsValues[batchIndex][rowIndex][columnIndex], decimal=4)
                        np.testing.assert_almost_equal(predictionsValues[batchIndex][rowIndex][0], predictionsValues[batchIndex][rowIndex][columnIndex], decimal=4)

            for batchIndex in range(len(predictionsValues)):
                for rowIndex in range(len(predictionsValues[batchIndex])):
                    for columnIndex in range(len(predictionsValues[batchIndex][rowIndex])):
                        prediction = predictionValuesValues[batchIndex][columnIndex][0]
                        target = targetValuesValues[batchIndex][0][rowIndex]
                        np.testing.assert_almost_equal(absDiffValues[batchIndex][columnIndex][rowIndex], abs(prediction - target), decimal=4)

            for batchIndex in range(len(absDiffValues)):
                for columnIndex in range(len(absDiffValues[batchIndex])):
                    for rowIndex in range(len(absDiffValues[batchIndex][columnIndex])):
                        absDiffValue = absDiffValues[batchIndex][columnIndex][rowIndex]
                        error = 0
                        if (absDiffValue <= kappa):
                            error = .5 * absDiffValue ** 2
                        else:
                            error = kappa * (absDiffValue - .5 * kappa)
                        np.testing.assert_almost_equal(totalErrorValues[batchIndex][columnIndex][rowIndex], error, decimal=4)

            for batchIndex in range(len(quantilePunishmentValues)):
                for columnIndex in range(len(quantilePunishmentValues[batchIndex])):
                    for rowIndex in range(len(quantilePunishmentValues[batchIndex][columnIndex])):
                        target = targetValuesValues[batchIndex][0][rowIndex]
                        prediction = predictionValuesValues[batchIndex][columnIndex][0]
                        relevantQuantile = quantileThresholdsValue[columnIndex]
                        punishment = (1 - relevantQuantile) if target < prediction else relevantQuantile
                        np.testing.assert_almost_equal(quantilePunishmentValues[batchIndex][columnIndex][rowIndex], punishment, decimal=4)
                        np.testing.assert_almost_equal(quantileRegressionLossValues[batchIndex][columnIndex][rowIndex], punishment * totalErrorValues[batchIndex][columnIndex][rowIndex], decimal=4)
    
    def testQuantileThresholds(self):
        graph = tf.Graph()
        numStateVariables = 1
        numActions = 1
        numQuantiles = 3
        embeddingDimension = 7
        with graph.as_default():
            sess = tf.Session()
            statePh = tf.placeholder(tf.float32, [None, numStateVariables], name="State_Placeholder")
            nextStatePh = tf.placeholder(tf.float32, [None, numStateVariables], name="NextState_Placeholder")
            actionsPh = tf.placeholder(tf.float32, [None, numActions], name="Actions_Placeholder")
            rewardsPh = tf.placeholder(tf.float32, [None, ], name="Rewards_Placeholder")
            terminalsPh = tf.placeholder(tf.float32, [None, ], name="Terminals_Placeholder")
            memoryPriorityPh = tf.placeholder(tf.float32, [None, ], name="MemoryPriority_Placeholder")
        batchSize = 4
        gamma = 0.99
        kappa = 1.0
        maxGradientNorm = 5.0 
        targetEntropy = -4.0
        entropyCoefficient = "auto"
        qNetwork = QNetwork(
            sess=sess,
            graph=graph,
            name="QNetwork_1_"+str(np.random.uniform(low=10000,high=99999)),
            numStateVariables=numStateVariables,
            numActions=numActions,
            networkSize=[16,16],
            numQuantiles=numQuantiles,
            gamma=gamma,
            kappa=kappa,
            learningRate=1e-4,
            showGraphs=False,
            statePh=statePh,
            nextStatePh=nextStatePh,
            actionsPh=actionsPh,
            rewardsPh=rewardsPh,
            terminalsPh=terminalsPh,
            memoryPriorityPh=memoryPriorityPh,
            maxGradientNorm=maxGradientNorm
        )
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            quantileThresholdsValues = sess.run(qNetwork.quantileThresholds)
            np.testing.assert_equal(quantileThresholdsValues, [.25, .5, .75])
