import numpy as np
from agent.Agent import Agent

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    actionScaling=2.0,
    policyNetworkSize=[64, 64],
    qNetworkSize=[64, 64],
    policyNetworkLearningRate=4e-3,
    qNetworkLearningRate=4e-3,
    entropyCoefficient="auto",
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=int(1e6),
    priorityExponent=0,
    batchSize=64,
    maxEpisodes=1024,
    trainSteps=1024,
    minStepsBeforeTraining=4096,
    rewardScaling=0.01,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=False,
    testSteps=1024,
    maxMinutes=600,
    targetEntropy=-1.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.02,
    varianceRegularizationConstant=0.01,
    randomStartSteps=100
)

print("Total Reward: "+str(agent.execute()))