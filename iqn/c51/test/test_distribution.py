# from c51.distribution import buildProjectOp
# from c51.distribution import buildTargetSupportOp
# import tensorflow as tf
# import unittest
# import numpy as np

# class TestProject(unittest.TestCase):
#     def test_type(self):
#         with tf.Session() as sess:
#             target = [4.0, 5.0, 6.0, 7.0, 8.0]
#             probTensor = tf.placeholder(tf.float32, [None, len(target)])
#             supportTensor = tf.placeholder(tf.float32, [None, len(target)])
#             projectOp = buildProjectOp(target, supportTensor, probTensor)
#             projectedSupport = sess.run(projectOp, feed_dict={
#                 probTensor: [
#                     [0.1, 0.6, 0.1, 0.1, 0.1],
#                     [0.1, 0.2, 0.5, 0.1, 0.1]
#                 ],
#                 supportTensor: [
#                     [0.0, 2.0, 4.0, 6.0, 8.0],
#                     [1.0, 3.0, 4.0, 5.0, 6.0]
#                 ]
#             })
#             np.testing.assert_almost_equal(projectedSupport, [
#                 [0.8, 0.0, 0.1, 0.0, 0.1],
#                 [0.8, 0.1, 0.1, 0.0, 0.0]
#             ])

# class TestBuildTargetSupports(unittest.TestCase):
#     def test_type(self):
#         with tf.Session() as sess:
#             support = [-2.0, 0.0, 2.0]
#             probTensor = tf.placeholder(tf.float32, [None, len(support)])
#             rewardTensor = tf.placeholder(tf.float32, [None,])
#             gammaTensor = tf.placeholder(tf.float32, [None,])
#             targetDistributionOp = buildTargetSupportOp(support, probTensor, rewardTensor, gammaTensor)
#             targetDist = sess.run(targetDistributionOp, feed_dict={
#                 probTensor: [
#                     [0.5, 0.25, 0.25],
#                     [0.25, 0.5, 0.25]
#                 ],
#                 rewardTensor: [1.0, 2.0],
#                 gammaTensor: [.5, .25]
#             })
#             np.testing.assert_almost_equal(targetDist, [
#                 [0.0, 1.0, 2.0],
#                 [1.5, 2.0, 2.5]
#             ])