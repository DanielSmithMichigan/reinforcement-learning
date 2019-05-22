import numpy as np
from agent.Agent import Agent

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    valueNetworkSize=[256, 256],
    entropyCoefficient=1.5e-3,
    valueNetworkLearningRate=4e-3,
    policyNetworkLearningRate=4e-3,
    qNetworkLearningRate=4e-3,
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=int(1e6),
    priorityExponent=0,
    batchSize=256,
    maxGradientNorm=5,
    maxEpisodes=1024,
    trainSteps=1024,
    minStepsBeforeTraining=1024,
    rewardScaling=0.1,
    actionScaling=4.0,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=False,
    weightRegularizationConstant=0.03,
    testSteps=1024,
    maxMinutes=1
)

print("Total Reward: "+str(agent.execute()))