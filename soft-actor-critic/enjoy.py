import numpy as np
from agent.Agent import Agent

rewardScaling = 10.0 ** -0.75

agent = Agent(
    name="agent_6073719",
    actionScaling=1.0,
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    policyNetworkLearningRate=3e-4,
    qNetworkLearningRate=3e-4,
    entropyCoefficient="auto",
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=int(5e6),
    priorityExponent=0.0,
    batchSize=256,
    maxEpisodes=0,
    trainSteps=1024,
    minStepsBeforeTraining=4096,
    rewardScaling=rewardScaling,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=True,
    showGraphs=False,
    saveModel=False,
    restoreModel=True,
    train=False,
    testSteps=1024,
    maxMinutes=60,
    targetEntropy=-4.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.0,
    varianceRegularizationConstant=0.0,
    randomStartSteps=0,
    gradientSteps=1,
    initialExtraNoise=0,
    extraNoiseDecay=0,
    evaluationEvery=250,
    numFinalEvaluations=10
)

print("Total Reward: "+str(agent.execute()))