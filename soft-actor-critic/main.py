import numpy as np
from agent.Agent import Agent

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
    maxMemoryLength=int(5e6),
    priorityExponent=0.0,
    batchSize=256,
    maxEpisodes=1024,
    trainSteps=1024,
    minStepsBeforeTraining=4096,
    rewardScaling=0.01,
    actionShift=0.0,
    stepsPerUpdate=1,
    render=False,
    showGraphs=True,
    saveModel=True,
    testSteps=1024,
    maxMinutes=60,
    targetEntropy=-4.0,
    maxGradientNorm=5.0,
    meanRegularizationConstant=0.0,
    varianceRegularizationConstant=0.0,
    randomStartSteps=10000,
    gradientSteps=1,
    initialExtraNoise=0.0,
    extraNoiseDecay=.99998,
    evaluationEvery=250
)

print("Total Reward: "+str(agent.execute()))