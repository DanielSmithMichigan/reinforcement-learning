import MySQLdb
import os
from agent.Agent import Agent
import numpy as np
import nevergrad as ng
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")
cur = db.cursor()

# experimentName = "bipedal-walker-priority-exponent"

try:
    agent = Agent(
        name=os.environ['AGENT_NAME'],
        actionScaling=1.0,
        policyNetworkSize=[256, 256],
        qNetworkSize=[256, 256],
        policyNetworkLearningRate=3e-4,
        qNetworkLearningRate=3e-4,
        entropyCoefficient="auto",
        tau=0.005,
        gamma=0.99,
        maxMemoryLength=5 * int(1e6),
        priorityExponent=0.0,
        batchSize=256,
        maxEpisodes=1024,
        trainSteps=1024,
        minStepsBeforeTraining=4096,
        rewardScaling=0.01,
        actionShift=0.0,
        stepsPerUpdate=4096,
        render=False,
        showGraphs=False,
        syncToS3=True,
        clearBufferAfterTraining=True,
        testSteps=1024,
        maxMinutes=60,
        targetEntropy=-4.0,
        maxGradientNorm=5.0,
        meanRegularizationConstant=0.0,
        varianceRegularizationConstant=0.0,
        randomStartSteps=10000,
        gradientSteps=16
    )

    result = agent.execute()
except:
    print("Error evaluating parameters")
    result = -20000
# cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}')".format(
#         experimentName,
#         priorityExponent,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         result
#     )
# )
# db.commit()
cur.close()
db.close()
