import MySQLdb
import os
from agent.Agent import Agent
import numpy as np
import nevergrad as ng
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")
cur = db.cursor()

experimentName = "soft-actor-critic-fixed-entropy"

entropyCoefficientArg = ng.instrumentation.variables.Gaussian(mean=-2, std=2)
learningRateArg = ng.instrumentation.variables.Gaussian(mean=-3, std=2)
rewardScalingArg = ng.instrumentation.variables.Gaussian(mean=-2, std=2)
actionScalingArg = ng.instrumentation.variables.Gaussian(mean=0.5, std=1)
weightRegularizationConstantArg = ng.instrumentation.variables.Gaussian(mean=-2, std=1)

instrumentation = ng.Instrumentation(entropyCoefficientArg, learningRateArg, rewardScalingArg, actionScalingArg, weightRegularizationConstantArg)
optimizer = ng.optimizers.registry["TBPSA"](instrumentation=instrumentation, budget=os.environ['BUDGET'])

cur.execute("select x1, x2, x3, x4, x5, y from experiments where label = '"+experimentName+"'")
result = cur.fetchall()
for row in result:
    candidate = optimizer.create_candidate.from_call(
        np.log10(float(row[0])),
        np.log10(float(row[1])),
        np.log10(float(row[2])),
        np.log10(float(row[3])),
        np.log10(float(row[4]))
    )
    optimizer.tell(candidate, -float(row[5]))

nextTest = optimizer.ask()

entropyCoefficient = 10 ** nextTest.args[0]
learningRate = 10 ** nextTest.args[1]
rewardScaling = 10 ** nextTest.args[2]
actionScaling = 10 ** nextTest.args[3]
weightRegularizationConstant = 10 ** nextTest.args[4]
result = -20000
try:
    agent = Agent(
        name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
        policyNetworkSize=[64, 64],
        qNetworkSize=[64, 64],
        valueNetworkSize=[64, 64],
        entropyCoefficient=entropyCoefficient,
        valueNetworkLearningRate=learningRate,
        policyNetworkLearningRate=learningRate,
        qNetworkLearningRate=learningRate,
        tau=0.005,
        gamma=0.99,
        maxMemoryLength=int(1e6),
        priorityExponent=0,
        batchSize=64,
        maxGradientNorm=5,
        maxEpisodes=1024,
        trainSteps=1024,
        minStepsBeforeTraining=4096,
        rewardScaling=rewardScaling,
        actionScaling=actionScaling,
        actionShift=0.0,
        stepsPerUpdate=1,
        render=False,
        showGraphs=False,
        meanRegularizationConstant=weightRegularizationConstant,
        varianceRegularizationConstant=weightRegularizationConstant,
        testSteps=1024,
        maxMinutes=15
    )

    result = agent.execute()
except:
    print("Error evaluating parameters")
    result = -20000
cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}')".format(
        experimentName,
        entropyCoefficient,
        learningRate,
        rewardScaling,
        actionScaling,
        weightRegularizationConstant,
        0,
        0,
        0,
        0,
        0,
        result
    )
)
db.commit()
cur.close()
db.close()