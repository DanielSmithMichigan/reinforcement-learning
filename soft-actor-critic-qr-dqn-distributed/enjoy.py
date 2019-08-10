import numpy as np
from agent.Agent import Agent

rewardScaling = 10.0 ** -0.75

agent = Agent(
    name="agent_4324310",
    actionScaling=1.0,
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    numQuantiles=60,
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
    minStepsBeforeTraining=4096,
    rewardScaling=(10.0 ** -0.75),
    actionShift=0.0,
    stepsPerUpdate=1,
    render=True,
    showGraphs=False,
    saveModel=False,
    saveModelToS3=False,
    restoreModel=True,
    train=False,
    testSteps=1024,
    maxMinutes=360,
    targetEntropy=-4.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.0,
    varianceRegularizationConstant=0.0,
    randomStartSteps=0,
    gradientSteps=1,
    initialExtraNoise=0,
    extraNoiseDecay=0,
    evaluationEvery=25,
    numFinalEvaluations=10,
    maxTrainSteps = 1000000
)

print("Total Reward: "+str(agent.execute()))