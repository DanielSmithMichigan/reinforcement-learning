import numpy as np
from agent.Agent import Agent

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    actionScaling=2.0,
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    valueNetworkSize=[256, 256],
    entropyCoefficient=0.0,
    valueNetworkLearningRate=3e-4,
    policyNetworkLearningRate=3e-4,
    qNetworkLearningRate=3e-4,
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=50000,
    priorityExponent=0,
    batchSize=64,
    maxGradientNorm=5,
    maxEpisodes=1024,
    trainSteps=1024,
    minStepsBeforeTraining=4096,
    rewardScaling=1.0,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=True,
    meanRegularizationConstant=0.01,
    varianceRegularizationConstant=0.04,
    testSteps=1024,
    maxMinutes=600,
    theta=0.15,
    sigma=.2,
    epsilonDecay=.99999,
    epsilonInitial=1.0
)

print("Total Reward: "+str(agent.execute()))