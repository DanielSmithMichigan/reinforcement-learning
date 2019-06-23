import MySQLdb
import os
import sys
from agent.Agent import Agent
import numpy as np
import nevergrad as ng
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")
cur = db.cursor()
results = [-20000]
agentName = "agent_"+str(np.random.randint(low=1000000,high=9999999))





experimentName = "bipedal-walker-priority-3"

rewardScaling = 10.0 ** -0.75
initialExtraNoise = np.random.uniform(0, 0.5)
extraNoiseDecay = 1.0 - (10 ** np.random.uniform(-7, -2))
maxMinutes = 180
priorityExponent = np.random.uniform() if np.random.uniform() < .8 else 0.0
    


try:
    agent = Agent(
        name=agentName,
        actionScaling=1.0,
        policyNetworkSize=[256, 256],
        qNetworkSize=[256, 256],
        policyNetworkLearningRate=3e-4,
        qNetworkLearningRate=3e-4,
        entropyCoefficient="auto",
        tau=0.005,
        gamma=0.99,
        maxMemoryLength=int(5e6),
        priorityExponent=priorityExponent,
        batchSize=64,
        maxEpisodes=4096,
        trainSteps=1024,
        minStepsBeforeTraining=4096,
        rewardScaling=rewardScaling,
        actionShift=0.0,
        stepsPerUpdate=1,
        render=False,
        showGraphs=False,
        saveModel=True,
        restoreModel=False,
        train=True,
        testSteps=1024,
        maxMinutes=maxMinutes,
        targetEntropy=-4.0,
        maxGradientNorm=5.0,
        meanRegularizationConstant=0.0,
        varianceRegularizationConstant=0.0,
        randomStartSteps=10000,
        gradientSteps=1,
        initialExtraNoise=0,
        extraNoiseDecay=0,
        evaluationEvery=50,
        numFinalEvaluations=10
    )

    results = agent.execute()
except:
    print("Unexpected error:", sys.exc_info()[0])
    results = [-20000]
for resultNum in range(len(results)):
    cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y, checkpoint, trainingSteps, agent_name) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}', '{12}', '{13}', '{14}')".format(
            experimentName,
            priorityExponent,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            results[resultNum][0],
            "checkpoint_"+str(resultNum),
            results[resultNum][1],
            agentName
        )
    )
    db.commit()
cur.close()
db.close()
