
from collections import deque
from . import constants
import tensorflow as tf
import numpy as np
import gym

class Actor:
    def __init__(
        self,
        policyNetwork,
        sess,
        memoryBuffer,
        maxEpisodeSteps,
        envName,
        actionScaling,
        deterministic
    ):
        self.policyNetwork = policyNetwork
        self.sess = sess
        self.memoryBuffer = memoryBuffer
        self.maxEpisodeSteps = maxEpisodeSteps
        self.env = gym.make(envName)
        self.episodeRewards = deque([], 400)
        self.actionScaling = actionScaling
        self.deterministic = deterministic
    def execute(
        self
    ):

    def episode(self, steps, evaluation, upload):
        state = self.env.reset()
        self.state = np.reshape(state, [self.numStateVariables,])
        self.totalEpisodeReward = 0
        done = False
        for stepNum in range(steps):
            done = self.goToNextState()
            if done:
                break
        if not done:
            self.goToNextState(endEarly=True)
        if evaluation:
            self.evaluations.append([
                self.totalEpisodeReward,
                self.trainingSteps
            ])
        self.episodeRewards.append(self.totalEpisodeReward)
        print("REWARD: "+str(self.totalEpisodeReward)+" FPS: "+str(fps))
    def goToNextState(self,endEarly=False):
        (
            rawAction,
            actionsChosen,
            qAssessment,
            deterministicAction,
            entropy
        ) = self.sess.run(
            self.actionOperations,
            feed_dict={
                self.statePh: [self.state]
            }
        )
        actionsChosen = actionsChosen[0] if not self.deterministic else deterministicAction[0]
        actionsChosen = actionsChosen * self.actionScaling
        self.actionsChosen.append(actionsChosen)
        self.entropyOverTime.append(entropy)
        nextState, reward, done, _ = self.env.step(actionsChosen)
        if endEarly:
            done = True
        nextState = np.reshape(nextState, [self.numStateVariables,])
        if (self.render):
            self.env.render()
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = self.state
        memoryEntry[constants.ACTION] = actionsChosen
        memoryEntry[constants.REWARD] = reward * self.rewardScaling
        memoryEntry[constants.NEXT_STATE] = nextState
        memoryEntry[constants.GAMMA] = self.gamma if not done else 0
        memoryEntry[constants.IS_TERMINAL] = done
        self.state = nextState
        self.memoryBuffer.add(memoryEntry)
        self.totalEpisodeReward = self.totalEpisodeReward + reward
        return done

    