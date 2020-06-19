import numpy as np
import gym

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib
import matplotlib.pyplot as plt

epsilon = 1e-8

def gaussian_probability(action, action_mean, log_action_std):
    return -0.5 * (((action - action_mean) / (tf.exp(log_action_std) + epsilon)) ** 2 + 2 * log_action_std + np.log(2 * np.pi))

env = gym.make("BipedalWalker-v3")
num_observations = 24
num_actions = 4
learning_rate = 0.01
population_size = 512
sigma = 0.1
max_steps = 1024
loss_surrogate_2_clip = 0.2
entropy_beta = 0.001
critic_discount = 0.5
gae_lambda = 0.95
gamma = 0.99

num_iterations = 1000

def compute_gae(rewards, is_terminals, values):
    # Need to append final value to values
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * is_terminals[step] - values[step]
        gae = delta + gamma * gae_lambda * is_terminals[step] * gae
        returns.insert(0, gae + values[step])
    return returns

layers = [
    40,
    40
]

num_iterations = 10000

overview = plt.figure()
lastNRewardsGraph = overview.add_subplot(1, 1, 1)

actions_ph = tf.placeholder(tf.float32, [None, num_actions], name="ActionsPlaceholder")
state_ph = tf.placeholder(tf.float32, [None, num_observations], name="StatePlaceholder")
randoms_placeholder = tf.placeholder(tf.float32, [None, num_actions], name="RandomsPlaceholder")
advantages_ph = tf.placeholder(tf.float32, [None, 1], name="AdvantagesPlaceholder")
returns_ph = tf.placeholder(tf.float32, [None, 1], name="ReturnsPlaceholder")
old_policy_log_prob_ph = tf.placeholder(tf.float32, [None, 1], name="OldPolicyLogProbPh")

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
        return value_prediction
    
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
            units=num_actions,
            activation=tf.nn.tanh,
            name="ActionMean"
        )
        log_action_std = tf.layers.dense(
            inputs=prev_layer,
            units=num_actions,
            activation=tf.nn.tanh,
            name="LogActionStd"
        )
        action_output = action_mean + randoms_placeholder * tf.exp(log_action_std)
        log_prob = tf.reduce_sum(gaussian_probability(actions_ph, action_mean, log_action_std) - tf.log(1 - actions_ph ** 2 + epsilon), axis=1)
        log_prob_action_output = tf.reduce_sum(gaussian_probability(action_output, action_mean, log_action_std) - tf.log(1 - action_output ** 2 + epsilon), axis=1)
        return action_output, log_prob_action_output, log_prob, action_mean
 

action_output, log_prob_action_output, log_prob, action_mean = build_policy_network()
value_prediction = build_value_network()

policy_ratio = tf.exp(log_prob - old_policy_log_prob_ph)

loss_surrogate_1 = policy_ratio * advantages_ph
loss_surrogate_2 = tf.clip_by_value(policy_ratio, 1.0 - loss_surrogate_2_clip, 1.0 + loss_surrogate_2_clip) * advantages_ph
actor_loss = -tf.reduce_mean(tf.minimum(loss_surrogate_1, loss_surrogate_2))
critic_loss = tf.reduce_mean((returns_ph - value_prediction) ** 2)
total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * tf.reduce_mean(-(log_prob * tf.log(log_prob + epsilon)))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

rewards = []
states = []
values = []
actions_chosen = []
value_predictions = []
log_probs = []
is_terminals = []
returns = []

for i in range(num_iterations):    
    print("Episode{}".format(i))
    state = env.reset()
    done = False
    total_reward = 0
    total_steps = 0

    while not done and total_steps < max_steps:
        [
            action_mean_val,
            action_chosen_val,
            value_prediction_val,
            log_prob_val
        ] = sess.run([
            action_mean,
            action_output,
            value_prediction,
            log_prob_action_output,
        ], feed_dict={
            state_ph: np.reshape(state, [1, num_observations]),
            randoms_placeholder: np.random.normal(0.0, 1.0, size=(1, num_actions))
        })

        next_state, reward, done, info = env.step(action_chosen_val[0])
        state = next_state
        total_reward += reward
        total_steps += 1

        rewards.append(reward)
        states.append(state)

        log_probs.append(log_prob_val)
        values.append(value_prediction_val)
        actions_chosen.append(action_chosen_val)
        is_terminals.append(done)
    
        value_prediction_val = sess.run(
            value_prediction,
            feed_dict={state_ph: np.reshape(state, [1, num_observations])}
        )
        
        returns = returns + compute_gae(rewards, is_terminals, values + [value_prediction_val])

