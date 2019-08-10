import tensorflow as tf
def buildTargetSupportOp(support, probabilities, rewards, gammas):
    supportSize = tf.shape(support)[0]
    batchSize = tf.shape(probabilities)[0]
    supports = tf.tile([support], [1, tf.shape(probabilities)[0]])
    supports = tf.reshape(supports, [batchSize, supportSize])
    supportSizedGammas = tf.reshape(gammas, [batchSize, 1])
    shrunkSupports = supports * supportSizedGammas
    reshapedRewards = tf.reshape(rewards, [batchSize, 1])
    shiftedSupports = tf.add(reshapedRewards, shrunkSupports)
    return shiftedSupports
def buildProjectOp(target, supports, probabilities):
    batchSize = tf.shape(supports)[0]
    supportSize = tf.shape(target)[0]
    projectionStructure = tf.tile(supports, [1, supportSize])
    projectionStructure = tf.reshape(projectionStructure, [-1, supportSize, supportSize])
    clippedProjectionStructure = tf.clip_by_value(projectionStructure, target[0], target[-1])
    tensorPerSupport = tf.tile([target], [1, batchSize])
    subtractableTensor = tf.reshape(tensorPerSupport, [batchSize, supportSize, 1])
    distanceFromZ = tf.abs(tf.subtract(clippedProjectionStructure, subtractableTensor))
    oppNormalized = 1 - (distanceFromZ / (target[1] - target[0]))
    clippedNormalized = tf.clip_by_value(oppNormalized, 0, 1)
    weightPerSupport = tf.tile(probabilities, [1, supportSize])
    weightPerSupport = tf.reshape(weightPerSupport, [-1, supportSize, supportSize])
    weightedNormalize = tf.multiply(weightPerSupport, clippedNormalized)
    return tf.reduce_sum(weightedNormalize, 2)
def buildTargets(support, probabilities, rewards, gammas):
    shiftedSupports = buildTargetSupportOp(support, probabilities, rewards, gammas)
    return buildProjectOp(support, shiftedSupports, probabilities)
