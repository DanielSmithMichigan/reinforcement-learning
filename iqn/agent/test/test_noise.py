# from agent import noise
# import random
# import tensorflow as tf
# import unittest
# import numpy as np
# import gym

# class TestNoisyLayer(unittest.TestCase):
#     def testOutputFives(self):
#         sess = tf.Session()
#         inputVar = tf.placeholder(dtype=tf.float32, shape=[None, 5])
#         layer = noise.noisy_dense_layer(inputVar, 5)
#         correctVal = tf.placeholder(dtype=tf.float32, shape=[None, 5])
#         loss = tf.losses.mean_squared_error(labels=correctVal, predictions=layer)
#         train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)
#         sess.run(tf.global_variables_initializer())
#         for i in range(1000):
#             _, lossVal = sess.run([train_op, loss], feed_dict={
#                 inputVar: np.random.uniform(low=-1, high=1, size=(2, 5)),
#                 correctVal: np.tile(5, (2, 5))
#             })
#         output = sess.run(layer, feed_dict={
#             inputVar: np.random.uniform(low=-1, high=1, size=(3, 5))
#         })
#         np.testing.assert_allclose(output, np.tile(5, (3, 5)), 1e-5)
#     def testNoise(self):
#         sess = tf.Session()
#         inputVar = tf.placeholder(dtype=tf.float32, shape=[None, 5])
#         layer, weightNoiseOut, biasNoiseOut = noise.noisy_dense_layer(inputVar, 5, debug=True)
#         correctVal = tf.placeholder(dtype=tf.float32, shape=[None, 5])
#         loss = tf.losses.mean_squared_error(labels=correctVal, predictions=layer)
#         train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)
#         sess.run(tf.global_variables_initializer())
#         # prevValue = sess.run(layer, feed_dict={
#         #     inputVar: np.tile(5, (3, 5))
#         # })
#         for i in range(5):
#             layerOut, w1, b = sess.run([layer, weightNoiseOut, biasNoiseOut], feed_dict={
#                 inputVar: np.tile(5, (3, 5))
#             })
#             # np.testing.assert_allclose(layer, prevValue, 1e-5)
#             # print(layer - prevValue)
#             # prevValue = layer



