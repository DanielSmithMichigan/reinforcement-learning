import numpy as np
from agent.Agent import Agent

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    valueNetworkSize=[256, 256],
    entropyCoefficient=1e-2,
    valueNetworkLearningRate=2e-4,
    policyNetworkLearningRate=2e-4,
    qNetworkLearningRate=2e-4,
    tau=0.001,
    gamma=.97,
    maxMemoryLength=int(1e6),
    priorityExponent=0,
    batchSize=64,
    maxGradientNorm=5,
    maxEpisodes=1024,
    maxSteps=1024,
    minStepsBeforeTraining=1024,
    rewardScaling=1e-1,
    actionScaling=2.0,
    stepsPerUpdate=1,
    render=True,
    theta=0.15,
    sigma=0.2
)

agent.execute()