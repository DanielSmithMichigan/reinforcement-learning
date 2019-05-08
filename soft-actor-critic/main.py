import numpy as np
from agent.Agent import Agent

agent = Agent(
    name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
    policyNetworkSize=[256, 256],
    qNetworkSize=[256, 256],
    valueNetworkSize=[256, 256],
    entropyCoefficient=0.1,
    valueNetworkLearningRate=3e-4,
    policyNetworkLearningRate=3e-4,
    qNetworkLearningRate=3e-4,
    tau=0.005,
    gamma=0.99,
    maxMemoryLength=int(1e6),
    priorityExponent=0,
    batchSize=256,
    maxGradientNorm=5,
    maxEpisodes=1024,
    maxSteps=1024,
    minStepsBeforeTraining=1024,
    rewardScaling=0.1,
    actionScaling=2.0,
    stepsPerUpdate=1,
    render=False,
    theta=0.15,
    sigma=0.2
)

agent.execute()