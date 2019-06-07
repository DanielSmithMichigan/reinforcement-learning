import numpy as np
from agent.Agent import Agent

priorityExponent = 10 ** np.random.uniform(-5, 0)

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    actionScaling=1.0,
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    policyNetworkLearningRate=3e-4,
    qNetworkLearningRate=3e-4,
    entropyCoefficient="auto",
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=5 * int(1e6),
    priorityExponent=priorityExponent,
    batchSize=256,
    maxEpisodes=1024,
    trainSteps=1024,
    minStepsBeforeTraining=10000,
    rewardScaling=0.01,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=False,
    save=False,
    testSteps=1024,
    maxMinutes=60,
    targetEntropy=-4.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.02,
    varianceRegularizationConstant=0.01,
    randomStartSteps=10000
)

print("Total Reward: "+str(agent.execute()))