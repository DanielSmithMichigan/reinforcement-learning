import gym
import numpy as np

class ParallelEnv:
    def __init__(
        self,
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
    ):
        self.num_envs = num_envs
        self.steps_per_training_iteration = steps_per_training_iteration
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.action_scaling = action_scaling
        self.reward_scaling = reward_scaling
        self.sess = sess
        self.action_mean = action_mean 
        self.action_output = action_output 
        self.value_prediction = value_prediction 
        self.log_prob = log_prob 
        self.entropy = entropy 
        self.log_action_std = log_action_std 
        self.state_ph = state_ph 
        self.randoms_placeholder = randoms_placeholder 
        self.envs = [ gym.make(env_name).unwrapped for i in range(num_envs) ]
        self.step_count = [ 0 for i in range(num_envs) ]
        self.training_reward_sum = [0 for i in range(num_envs) ]
        self.current_state = [ np.append(self.envs[i].reset(), [0] ) for i in range(num_envs) ]
        self.rewards = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration), dtype="float64")
        self.states = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration + 1, self.envs[0].observation_space.shape[0] + 1), dtype="float64")
        self.actions_chosen = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration, self.envs[0].action_space.shape[0]), dtype="float64")
        self.value_predictions = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration + 1), dtype="float64")
        self.log_probs = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration), dtype="float64")
        self.first_steps = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration + 1), dtype="float64")
        self.entropies = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration), dtype="float64")
        self.max_episode_steps = max_episode_steps
    
    def calculate_returns_advantages(self):
        ones = np.ones((self.num_envs), dtype="float64")
        first_steps = np.array(self.first_steps, dtype="float64")
        returns = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration), dtype="float64")
        advantages = np.ndarray(shape=(self.num_envs, self.steps_per_training_iteration), dtype="float64")
        gae = 0
        for step in reversed(range(self.steps_per_training_iteration)):
            step_first_steps = first_steps[ : , step + 1]
            step_nonterminal = ones - step_first_steps
            step_rewards = self.rewards[ : , step]
            step_values = self.value_predictions[ : , step]
            step_next_values = self.value_predictions[ : , step + 1]
            delta = step_rewards + self.gamma * step_next_values * step_nonterminal - step_values
            gae = delta + self.gamma * self.gae_lambda * step_nonterminal * gae
            returns[ : , step ] = gae
            advantages[ : , step ] = gae - step_values

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages

    def step(self):
        for step_num in range(self.steps_per_training_iteration):
            [
                action_mean_vals,
                action_chosen_vals,
                value_prediction_vals,
                log_prob_vals,
                entropy_vals,
                log_action_std_vals,
            ] = self.sess.run([
                self.action_mean,
                self.action_output,
                self.value_prediction,
                self.log_prob,
                self.entropy,
                self.log_action_std,
            ], feed_dict={
                self.state_ph: self.current_state,
                self.randoms_placeholder: np.random.normal(0.0, 1.0, size=(self.num_envs, self.envs[0].action_space.shape[0]))
            })

            self.entropies[ : , step_num ] = entropy_vals
            self.value_predictions[ : , step_num ] = value_prediction_vals
            self.actions_chosen[ : , step_num, : ] = action_chosen_vals
            self.log_probs[ : , step_num ] = log_prob_vals
            self.states[ : , step_num ] = self.current_state

            action_chosen_vals = self.action_scaling * np.tanh(action_chosen_vals)

            for env_num in range(self.num_envs):
                next_state, reward, terminal, _ = self.envs[env_num].step(action_chosen_vals[env_num])

                self.rewards[env_num][step_num] = reward * self.reward_scaling
                self.first_steps[env_num][step_num] = self.step_count[env_num] == 0

                state = next_state

                if terminal or self.step_count[env_num] >= self.max_episode_steps:
                    self.step_count[env_num] = 0
                    state = self.envs[env_num].reset()
                else:
                    self.step_count[env_num] += 1

                state = np.reshape(state, [self.envs[0].observation_space.shape[0]])
                state = np.append(state, [self.step_count[env_num] / self.max_episode_steps])

                self.current_state[env_num] = state
                self.states[env_num][step_num + 1] = state
                self.first_steps[env_num][step_num + 1] = self.step_count[env_num] == 0

        value_prediction_vals = self.sess.run(
            self.value_prediction,
            feed_dict={ self.state_ph: self.states[ : , -1 ] }
        )

        self.value_predictions[ : , -1 ] = np.reshape(value_prediction_vals, [self.num_envs])
        
        self.returns, self.advantages = self.calculate_returns_advantages()

        training_iteration_states = self.states[ : , 0 : self.steps_per_training_iteration , : ]
        training_iteration_actions_chosen = self.actions_chosen
        training_iteration_value_predictions = self.value_predictions[ : , 0 : self.steps_per_training_iteration]
        training_iteration_log_probs = self.log_probs
        training_iteration_returns = self.returns
        training_iteration_advantages = self.advantages

        return [
            np.reshape(training_iteration_states, [-1, self.envs[0].observation_space.shape[0] + 1]),
            np.reshape(training_iteration_actions_chosen, [-1, self.envs[0].action_space.shape[0]]),
            np.reshape(training_iteration_value_predictions, [-1]),
            np.reshape(training_iteration_log_probs, [-1]),
            np.reshape(training_iteration_returns, [-1]),
            np.reshape(training_iteration_advantages, [-1]),
        ]
            