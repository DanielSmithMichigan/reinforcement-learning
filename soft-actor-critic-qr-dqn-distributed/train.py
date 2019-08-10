from agent.Agent import Agent
import numpy as np
import nevergrad as ng
import tensorflow as tf
import gym

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    numStateVariables=24,
    numActionVariables=4,
    envName='BipedalWalker-v2',
    actionScaling=1.0,
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    numQuantiles=16,
    policyNetworkLearningRate=3e-4,
    qNetworkLearningRate=3e-4,
    entropyCoefficient="auto",
    tau=0.005,
    gamma=0.99,
    kappa=1.0,
    maxMemoryLength=int(5e6),
    priorityExponent=0.0,
    batchSize=64,
    maxEpisodes=4096,
    trainSteps=1024,
    maxTrainSteps=6000000,
    minStepsBeforeTraining=4096,
    rewardScaling=(10.0 ** -0.75),
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=False,
    saveModel=False,
    saveModelToS3=False,
    restoreModel=False,
    train=True,
    testSteps=1024,
    maxMinutes=360,
    targetEntropy=-4.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.0,
    varianceRegularizationConstant=0.0,
    randomStartSteps=10000,
    gradientSteps=1,
    initialExtraNoise=0,
    extraNoiseDecay=0,
    evaluationEvery=25,
    numFinalEvaluations=10
)

results = agent.execute()
