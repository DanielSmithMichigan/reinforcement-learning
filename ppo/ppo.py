import numpy as np
import gym

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import math

import random

# from ParallelEnv2 import SubprocVecEnv
from ParallelEnv import ParallelEnv
plt.ion()

overview = plt.figure()
loss_graph = overview.add_subplot(2, 1, 1)
critic_graph = overview.add_subplot(2, 1, 2)

prediction_pair_figure = plt.figure()
prediction_pair_graph = prediction_pair_figure.add_subplot(1, 1, 1)

entropy_figure = plt.figure()
entropy_graph = entropy_figure.add_subplot(1, 1, 1)

epsilon = 1e-8

def neg_log_prob_tensor(actions_chosen, action_mean, log_action_std):
    std = tf.exp(log_action_std)
    return 0.5 * tf.reduce_sum(tf.square((actions_chosen - action_mean) / std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(actions_chosen)[-1], tf.float32) \
               + tf.reduce_sum(log_action_std, axis=-1)

def log_prob_tensor(actions_chosen, action_mean, log_action_std):
    return - neg_log_prob_tensor(actions_chosen, action_mean, log_action_std)

    

# def entropy(self):
#     return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

def entropy_tensor(log_action_std):
    return tf.reduce_sum(log_action_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

env_name = "Pendulum-v0"
test_env = gym.make(env_name)
action_scaling = 2.0
learning_rate = 2e-4
loss_surrogate_2_clip = 0.2
entropy_coefficient = 0.04
critic_discount = 0.5
gae_lambda = 0.95
gamma = 0.99
num_envs = 8
steps_per_training_iteration = 512
steps_per_evaluation = 1024
max_episode_steps = 2048
num_training_iterations = 10000
max_gradient_norm = 5
batch_size = 64
training_epochs = 10
target_entropy = -4.0
reward_scaling = 1e-2
IMAGE_SIZE = 200

layers = [
    64,
    64
]

actions_ph = tf.placeholder(tf.float32, [None,  test_env.action_space.shape[0]], name="ActionsPlaceholder")
state_ph = tf.placeholder(tf.float32, [None,  test_env.observation_space.shape[0] + 1], name="StatePlaceholder")
randoms_placeholder = tf.placeholder(tf.float32, [None,  test_env.action_space.shape[0]], name="RandomsPlaceholder")
advantages_ph = tf.placeholder(tf.float32, [None,], name="AdvantagesPlaceholder")
returns_ph = tf.placeholder(tf.float32, [None,], name="ReturnsPlaceholder")
old_action_old_log_prob_ph = tf.placeholder(tf.float32, [None,], name="OldPolicyLogProbPh")

def build_value_network():
    with tf.variable_scope("ValueNetwork"):
        prev_layer = state_ph
        for i in range(len(layers)):
            prev_layer = tf.layers.dense(
                inputs=prev_layer,
                units=layers[i],
                activation=tf.nn.leaky_relu,
                name="Dense" + str(i)
            )
        value_prediction = tf.layers.dense(
            inputs=prev_layer,
            units=1,
            activation=None,
            name="ValuePrediction"
        )
        return tf.reshape(value_prediction, [-1])
    
def build_policy_network():
    with tf.variable_scope("PolicyNetwork"):
        prev_layer = state_ph
        for i in range(len(layers)):
            prev_layer = tf.layers.dense(
                inputs=prev_layer,
                units=layers[i],
                activation=tf.nn.leaky_relu,
                name="Dense" + str(i)
            )
        action_mean = tf.layers.dense(
            inputs=prev_layer,
            units=test_env.action_space.shape[0],
            activation=None,
            name="ActionMean"
        )
        log_action_std = tf.layers.dense(
            inputs=prev_layer,
            units=test_env.action_space.shape[0],
            activation=None,
            name="LogActionStd"
        )
        action_chosen = action_mean + randoms_placeholder * tf.exp(log_action_std)
        log_prob = log_prob_tensor(action_chosen, action_mean, log_action_std)
        old_action_new_log_prob = log_prob_tensor(actions_ph , action_mean, log_action_std)
        entropy = entropy_tensor(log_action_std)
        return action_chosen, log_prob, entropy, old_action_new_log_prob, action_mean, log_action_std
 

action_output, log_prob, entropy, old_action_new_log_prob, action_mean, log_action_std = build_policy_network()
value_prediction = build_value_network()


policy_ratio = tf.exp(old_action_new_log_prob - old_action_old_log_prob_ph)


loss_surrogate_1 = policy_ratio * advantages_ph
loss_surrogate_2 = tf.clip_by_value(policy_ratio, 1.0 - loss_surrogate_2_clip, 1.0 + loss_surrogate_2_clip) * advantages_ph
min_loss_surrogates = tf.minimum(loss_surrogate_1, loss_surrogate_2)
actor_loss = -tf.reduce_mean(min_loss_surrogates)
batch_critic_loss = (returns_ph - value_prediction) ** 2
critic_loss = tf.reduce_mean(batch_critic_loss)
entropy_loss = tf.reduce_mean(entropy)

total_loss = critic_discount * critic_loss + actor_loss - entropy_coefficient * entropy_loss
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients, variables = zip(
    *optimizer.compute_gradients(
        total_loss,
        var_list=tf.trainable_variables()
    )
)
gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
training_operation = optimizer.apply_gradients(
    zip(gradients, variables)
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

policyFigure = plt.figure()
policyFigure.suptitle("Policy")
policyGraph = policyFigure.add_subplot(1, 1, 1)
divider = make_axes_locatable(policyGraph)
policyColorBar = divider.append_axes("right", size="7%", pad="2%")

def updateAssessmentGraphs():
    states = []
    imageRadius = IMAGE_SIZE / 2
    for xImg in range(IMAGE_SIZE):
        for yImg in range(IMAGE_SIZE):
            x = xImg - imageRadius
            y = yImg - imageRadius
            v = np.clip(math.sqrt(x * x + y * y) * 16 / imageRadius, 0, 16)
            v = v - 8

            theta = None
            if x < 0:
                theta = math.atan(y / x) + math.pi
            elif x == 0 and y > 0:
                theta = math.pi
            elif x == 0 and y < 0:
                theta = -math.pi
            elif x == 0 and y == 0:
                theta = 0
            elif x > 0 and y < 0:
                theta = math.atan(y / x) + math.pi + math.pi
            else:
                theta = math.atan(y / x)
            states.append([math.cos(theta), math.sin(theta), v])
    (
        action_mean_val
    ) = sess.run([
        action_mean
    ], feed_dict={
        state_ph: states
    })

    actionsChosenImg = np.reshape(action_mean_val, [IMAGE_SIZE, IMAGE_SIZE])

    policyGraph.cla()
    policyColorBar.cla()
    ax=policyGraph.imshow(actionsChosenImg)
    colorbar(ax, cax=policyColorBar)
    policyFigure.canvas.draw()

total_loss_over_time = []
prediction_pairs = []
loss_surrogate_1_over_time = []
loss_surrogate_2_over_time = []
actor_loss_over_time = []
critic_loss_over_time = []
entropy_loss_over_time = []
entropy_over_time = []

env = ParallelEnv(
    num_envs,
    env_name,
    steps_per_training_iteration,
    gamma,
    gae_lambda,
    action_scaling,
    reward_scaling,
    sess,
    action_mean,
    action_output,
    value_prediction,
    log_prob,
    entropy,
    log_action_std,
    state_ph,
    randoms_placeholder,
    max_episode_steps,
)

for training_iteration_idx in range(num_training_iterations):

    [
        training_iteration_states,
        training_iteration_actions_chosen,
        training_iteration_value_predictions,
        training_iteration_log_probs,
        training_iteration_returns,
        training_iteration_advantages,
    ] = env.step()

    for epoch_key in range(training_epochs):

        indices = [*range(len(training_iteration_states))]
        random.shuffle(indices)
        batches = np.array_split(indices, len(indices) // batch_size)

        for batch_key in range(len(batches)):
            batch_idx = batches[batch_key]
            batch_advantages = np.reshape(training_iteration_advantages[batch_idx], [-1])
            batch_returns = np.array(training_iteration_returns)[batch_idx]
            batch_returns = np.reshape(batch_returns, [-1])
            batch_states = np.array(training_iteration_states)[batch_idx]
            batch_old_policy_log_probs = np.reshape(training_iteration_log_probs[batch_idx], [-1])
            batch_actions = training_iteration_actions_chosen[batch_idx]
            batch_randoms = np.random.normal(0.0, 1.0, size=(batch_size, test_env.action_space.shape[0]))
            # print("Training Operation {}:{}".format(epoch_key, batch_key))
            [
                _,
                total_loss_val,
                policy_ratio_val,
                loss_surrogate_1_val,
                loss_surrogate_2_val,
                actor_loss_val,
                critic_loss_val,
                entropy_loss_val,
                value_prediction_vals
            ] = sess.run([
                training_operation,
                total_loss,
                policy_ratio,
                loss_surrogate_1,
                loss_surrogate_2,
                actor_loss,
                critic_loss,
                entropy_loss,
                value_prediction
            ], feed_dict={
                    advantages_ph: batch_advantages,
                    returns_ph: batch_returns,
                    state_ph: batch_states,
                    old_action_old_log_prob_ph: batch_old_policy_log_probs,
                    actions_ph: batch_actions,
                    randoms_placeholder: batch_randoms
                }
            )

            # print('''
            #     Total loss: {}
            #     Actor Loss: {}
            #     Critic Loss: {}
            #     Loss Surrogate 1 {}
            #     Loss Surrogate 2: {}
            #     Policy Ratio: {}
            #     Entropy Loss: {}
            # '''.format(
            #     total_loss_val,
            #     actor_loss_val,
            #     critic_loss_val,
            #     np.mean(loss_surrogate_1_val),
            #     np.mean(loss_surrogate_2_val),
            #     np.mean(policy_ratio_val),
            #     entropy_loss_val
            # )
            

            if (batch_key % 25) == 0:
                total_loss_over_time = total_loss_over_time + [total_loss_val]
                loss_surrogate_1_over_time.append(np.mean(loss_surrogate_1_val))
                loss_surrogate_2_over_time.append(np.mean(loss_surrogate_2_val))
                entropy_loss_over_time.append(entropy_loss_val)

                actor_loss_over_time.append(actor_loss_val)
                critic_loss_over_time.append(critic_loss_val)

                critic_graph.cla()
                critic_graph.plot(critic_loss_over_time, label="Critic Loss")
                prediction_pair_graph.cla()
                prediction_pair_graph.scatter(value_prediction_vals / reward_scaling, batch_returns / reward_scaling, label="Predictions")
                prediction_pair_graph.set_xlim(-100, 100)
                prediction_pair_graph.set_ylim(-100, 100)
                entropy_graph.cla()
                entropy_graph.plot(entropy_over_time, label="Entropy Over Time")

                loss_graph.cla()
                # loss_graph.plot(total_loss_over_time, label="Total Loss")
                loss_graph.plot(loss_surrogate_1_over_time, label="Loss Surrogate 1")
                loss_graph.plot(loss_surrogate_2_over_time, label="Loss Surrogate 2")
                loss_graph.plot(actor_loss_over_time, label="Actor Loss")
                loss_graph.plot(critic_loss_over_time, label="Critic Loss")
                loss_graph.plot(entropy_loss_over_time, label="Entropy Loss")
                loss_graph.legend(loc=2)

                overview.canvas.draw()
                prediction_pair_figure.canvas.draw()
                entropy_figure.canvas.draw()

                # updateAssessmentGraphs()
                plt.pause(0.01)

    state = test_env.reset()
    state = np.reshape(state, [test_env.observation_space.shape[0]])
    state = np.append(state, [0])

    test_env.render()

    done = False
    total_rewards = 0

    for i in range(steps_per_evaluation):
        action_mean_val = sess.run(action_mean, feed_dict={ state_ph: [ state ] })
        action_mean_val = action_mean_val[0]
        action_mean_val = action_scaling * np.tanh(action_mean_val)

        state, reward, done, _ = test_env.step(action_mean_val)
        test_env.render()
        state = np.reshape(state, [test_env.observation_space.shape[0]])
        state = np.append(state, [i / steps_per_evaluation])

        total_rewards += reward
    print("total_rewards: {}".format(total_rewards))

