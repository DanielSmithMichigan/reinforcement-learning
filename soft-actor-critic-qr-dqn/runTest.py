import MySQLdb
import os
import sys
from agent.Agent import Agent
import numpy as np
import tensorflow as tf
import gym
import math
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")
cur = db.cursor()
results = [-20000]
agentName = "agent_"+str(np.random.randint(low=1000000,high=9999999))

experimentName = "qr-dqn-nStep"
# experimentName = "qr-dqn-actor-critic"

nStep = np.random.randint(low=1, high=12)


try:
    agent = Agent(
        name=agentName,
        actionScaling=1.0,
        policyNetworkSize=[256, 256],
        qNetworkSize=[256, 256],
        numQuantiles=16,
        policyNetworkLearningRate=3e-4,
        qNetworkLearningRate=3e-4,
        entropyCoefficient="auto",
        tau=0.005,
        gamma=0.99,
        kappa=1.0,
        maxMemoryLength=int(5e6),
        priorityExponent=0.0,
        batchSize=64,
        nStep=nStep,
        maxEpisodes=4096,
        trainSteps=1024,
        maxTrainSteps=6000000,
        minStepsBeforeTraining=40000,
        rewardScaling=(10.0 ** -0.75),
        actionShift=0.0,
        stepsPerUpdate=1,
        render=False,
        showGraphs=False,
        saveModel=False,
        saveModelToS3=False,
        restoreModel=False,
        train=True,
        testSteps=1024,
        maxMinutes=180,
        targetEntropy=-4.0,
        maxGradientNorm=5.0,
        meanRegularizationConstant=0.0,
        varianceRegularizationConstant=0.0,
        randomStartSteps=40000,
        gradientSteps=1,
        initialExtraNoise=0,
        extraNoiseDecay=0,
        evaluationEvery=25,
        numFinalEvaluations=10
    )

    results = agent.execute()
except:
    print("Unexpected error:", sys.exc_info()[0])
    results = [-20000]
for resultNum in range(len(results)):
    cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y, checkpoint, trainingSteps, agent_name) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}', '{12}', '{13}', '{14}')".format(
            experimentName,
            nStep,
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
