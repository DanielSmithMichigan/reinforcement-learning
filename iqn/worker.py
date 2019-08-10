from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym

def objective(args):
    NUM_TESTS_FOR_NOISE = 1
    env = gym.make('LunarLander-v2')
    learningRate = args
    # numIntermediateLayers = int(numIntermediateLayers)
    # intermediateLayerSize = int(intermediateLayerSize)
    # finalLayerSize = int(finalLayerSize)
    # layers = []
    # for i in range(numIntermediateLayers):
    #     layers.append(intermediateLayerSize)
    # layers.append(finalLayerSize)
    # print("Layers: ",layers)
    # print("Priority: ",priorityExponent)
    # print("LR: ",learningRate)
    totalResult = 0
    for i in range(NUM_TESTS_FOR_NOISE):
        sess = tf.Session()
        a = Agent(
            sess=sess,
            env=env,
            numAvailableActions=4,
            numObservations=8,
            rewardsMovingAverageSampleLength=20,
            gamma=1,
            nStepUpdate=1,
            includeIntermediatePairs=False,
            maxRunningMinutes=30,

            # test parameters
            episodesPerTest=1,
            numTestPeriods=40000,
            numTestsPerTestPeriod=30,
            episodeStepLimit=1024,
            intermediateTests=False,

            render=False,
            showGraph=False,

            # hyperparameters
            valueMin=-400.0,
            valueMax=300.0,
            numAtoms=14,
            maxMemoryLength=100000,
            batchSize=256,
            networkSize=[128, 128, 256],
            learningRate=learningRate,
            priorityExponent=0,
            epsilonInitial = 2,
            epsilonDecay = .9987,
            minFramesForTraining = 2048,
            noisyLayers = False,
            maxGradientNorm = 4,
            minExploration = .15,
        )
        testResults = np.array(a.execute())
        performance = np.mean(testResults[np.argpartition(-testResults,range(4))[:4]])
        totalResult = totalResult + performance
    print(str(learningRate)+","+str(performance))
    return -totalResult