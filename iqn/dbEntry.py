import MySQLdb
import os
from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "iqn-embedding-sawtooth"
embeddingDimension = np.random.randint(low=4, high=16)
env = gym.make('LunarLander-v2')
sess = tf.Session()
rewardScaling = pow(10, np.random.uniform(low=-2, high=-1))
a = Agent(
    sess=sess,
    env=env,
    numAvailableActions=4,
    numObservations=8,
    rewardsMovingAverageSampleLength=200,
    gamma=.99,
    nStepUpdate=1,
    includeIntermediatePairs=False,

    # test parameters
    episodesPerTest=25,
    stepsPerTrainingPeriod=4,
    numTestPeriods=1024,
    numTestsPerTestPeriod=20,
    maxRunningMinutes=60 * 4,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=False,
    saveModel=False,
    loadModel=False,
    disableRandomActions=False,
    disableTraining=False,
    # agentName="agent_842763505",

    # hyperparameters
    rewardScaling=rewardScaling,
    nStepReturns=1,
    maxMemoryLength=int(1e6),
    batchSize=64,
    learningRate=6.25e-4,
    priorityExponent= 0,
    epsilonInitial = 1,
    epsilonDecay = .997,
    minExploration = .01,
    maxExploration = 1.0,
    minFramesForTraining = 2048,
    maxGradientNorm = 5,
    preNetworkSize = [128,128],
    postNetworkSize = [256],
    numQuantiles = 8,
    embeddingDimension = embeddingDimension,
    kappa = 1.0,
    trainingIterations = 3,
    tau = 0.001
)
performance = a.execute()[0]
cur = db.cursor()
cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y, checkpoint, trainingSteps, agent_name) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}', '{12}', '{13}', '{14}')".format(
        experimentName,
        embeddingDimension,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        performance,
        "checkpoint_0",
        1024,
        "no_agent_name"
    )
)
db.commit()
cur.close()
db.close()




















