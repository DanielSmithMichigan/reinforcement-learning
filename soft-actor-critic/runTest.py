import MySQLdb
import os
from agent.Agent import Agent
import numpy as np
import nevergrad as ng
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "soft-actor-critic-general"
env = gym.make('LunarLander-v2')

entropyCoefficientArg = Gaussian(mean=-2, std=2)
learningRateArg = Gaussian(mean=-3, std=2)
rewardScalingArg = Gaussian(mean=-3, std=2)
actionScalingArg = Gaussian(mean=-1, std=1)
weightRegularizationConstantArg = Gaussian(mean=-2, std=1)

value = Gaussian(mean=-5000, std=2000)

instrumentation = ng.Instrumentation(entropyCoefficientArg, learningRateArg, rewardScalingArg, actionScalingArg, weightRegularizationConstantArg, value=value)
optimizer = ng.optimizers.registry["TBPSA"](instrumentation=instrumentation, budget=200)

db.query("select x1, x2, x3, x4, x5, y from experiments where experimentName = '"+experimentName+"'")
result = db.store_result()
for i in range(1000):
    row = result.fetch_row(maxrows=0, how=1)
    print(row)
# candidate = optimizer.create_candidate.from_call(*args, **kwargs)

# entropyCoefficient = 10 ** np.random.uniform(-5, 0)
# learningRate = 10 ** np.random.uniform(-5, -2)
# rewardScaling = 10 ** np.random.uniform(-4, 1)
# actionScaling = 10 ** np.random.uniform(-1, 1)
# weightRegularizationConstant = 10 ** np.random.uniform(-3,-1)

# agent = Agent(
#     name="agent_"+str(np.random.randint(low=1000000,high=9999999)),
#     policyNetworkSize=[64, 64],
#     qNetworkSize=[64, 64],
#     valueNetworkSize=[64, 64],
#     entropyCoefficient=entropyCoefficient,
#     valueNetworkLearningRate=learningRate,
#     policyNetworkLearningRate=learningRate,
#     qNetworkLearningRate=learningRate,
#     tau=0.005,
#     gamma=0.99,
#     maxMemoryLength=int(1e6),
#     priorityExponent=0,
#     batchSize=256,
#     maxGradientNorm=5,
#     maxEpisodes=1024,
#     trainSteps=1024,
#     minStepsBeforeTraining=4096,
#     rewardScaling=rewardScaling,
#     actionScaling=actionScaling,
#     actionShift=0.0,
#     stepsPerUpdate=1,
#     render=False,
#     showGraphs=False,
#     weightRegularizationConstant=weightRegularizationConstant,
#     testSteps=1024,
#     maxMinutes=5
# )

# result = agent.execute()
# cur = db.cursor()
# cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}')".format(
#         experimentName,
#         entropyCoefficient,
#         learningRate,
#         rewardScaling,
#         actionScaling,
#         weightRegularizationConstant,
#         0,
#         0,
#         0,
#         0,
#         0,
#         result
#     )
# )
# db.commit()
# cur.close()
# db.close()