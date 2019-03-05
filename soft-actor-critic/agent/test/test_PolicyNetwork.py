import tensorflow as tf
import numpy as np
import unittest
from agent.PolicyNetwork import PolicyNetwork
from agent import constants
sess = tf.Session()
policyNetwork = PolicyNetwork(
    sess=sess,
    name="PolicyNetwork",
    numStateVariables=1,
    numActions=1,
    networkSize=[256,256],
    entropyCoefficient=1.0,
    learningRate=2e-4,
    maxGradientNorm=5.0,
    batchSize=64,
    theta=0.15,
    sigma=0.2
)
q = type('', (), {})()
q.actionsPh = tf.placeholder(tf.float32, [None, 1], name="Actions")
q.statePh = tf.placeholder(tf.float32, [None, 1], name="State")
q.qValue = -tf.abs(.25 - q.actionsPh)
policyNetwork.setQNetwork(q)
policyNetwork.buildTrainingOperation()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    memories = []
    for j in range(64):
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = [np.random.random()]
        memoryEntry[constants.ACTION] = [np.random.random()]
        memoryEntry[constants.REWARD] = [np.random.random()]
        memoryEntry[constants.NEXT_STATE] = [np.random.random()]
        memoryEntry[constants.GAMMA] = 0
        memoryEntry[constants.IS_TERMINAL] = False
        memories.append(memoryEntry)
    policyNetwork.trainAgainst(memories)
    action = policyNetwork.getAction([np.random.random()])
    qValue = sess.run(q.qValue, feed_dict={
        q.actionsPh: [action],
        q.statePh: np.reshape([np.random.random()], [-1, 1])
    })
    print("action: ",action[0]," qValue: ",qValue[0][0])


# class TestPolicyNetwork(unittest.TestCase):
#     def testTrainWithSimpleQ(self):
#         a = 1