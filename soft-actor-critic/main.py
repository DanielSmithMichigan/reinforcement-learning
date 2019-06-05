import numpy as np
from agent.Agent import Agent

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    actionScaling=1.0,
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    valueNetworkSize=[256, 256],
    valueNetworkLearningRate=4e-3,
    policyNetworkLearningRate=4e-3,
    qNetworkLearningRate=4e-3,
    entropyCoefficient="auto",
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=int(1e6),
    priorityExponent=0,
    batchSize=256,
    maxEpisodes=1024,
    trainSteps=1024,
    minStepsBeforeTraining=4096,
    rewardScaling=1.0,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=True,
    testSteps=1024,
    maxMinutes=600,
    targetEntropy=-4.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.02,
    varianceRegularizationConstant=0.07
)

print("Total Reward: "+str(agent.execute()))