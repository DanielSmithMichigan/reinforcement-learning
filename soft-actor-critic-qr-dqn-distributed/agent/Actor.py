import tensorflow as tf
import numpy as np
import gym

from . import constants
from . import util

import time

class Actor:
    def __init__(
        self,
        threadName,
        deterministic,
        envName,
        actionOperations,
        actionScaling,
        statePh,
        memoryBuffer,
        noiseScaling,
        numActions,
        numStateVariables,
        randomStartSteps,
        gamma,
        rewardScaling,
        episodeSteps,
        maxEpisodes,
        sess
    ):
        self.threadName = threadName
        self.env = gym.make(envName)
        self.sess = sess
        self.actionOperations = actionOperations
        self.statePh = statePh
        self.memoryBuffer = memoryBuffer
        self.deterministic = deterministic
        self.actionScaling = actionScaling
        self.noiseScaling = noiseScaling
        self.numActions = numActions
        self.numStateVariables = numStateVariables
        self.randomStartSteps = randomStartSteps
        self.gamma = gamma
        self.rewardScaling = rewardScaling
        self.episodeSteps = episodeSteps
        self.maxEpisodes = maxEpisodes
        self.globalStep = 0
        self.lastGlobalStep = 0
        self.lastTime = time.time()
        self.episodeRewards = []
        self.fpsOverTime = []
    def updateFps(self):
        newTime = time.time()
        timeSpent = newTime - self.lastTime
        framesRendered = self.globalStep - self.lastGlobalStep 
        fps = framesRendered / timeSpent
        self.lastGlobalStep = self.globalStep
        self.lastTime = newTime
        self.fpsOverTime.append(fps)
        return fps
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
        if not self.deterministic:
            actionsChosen += np.random.normal(loc=0.0, scale=self.noiseScaling, size=(self.numActions,))
        if self.globalStep < self.randomStartSteps:
            actionsChosen = self.env.action_space.sample()
        nextState, reward, done, info = self.env.step(actionsChosen)
        if endEarly:
            done = True
        nextState = np.reshape(nextState, [self.numStateVariables,])
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = self.state
        memoryEntry[constants.ACTION] = actionsChosen
        memoryEntry[constants.REWARD] = reward * self.rewardScaling
        memoryEntry[constants.NEXT_STATE] = nextState
        memoryEntry[constants.GAMMA] = self.gamma if not done else 0
        memoryEntry[constants.IS_TERMINAL] = done
        self.state = nextState
        self.memoryBuffer.add(memoryEntry)
        self.globalStep += 1
        self.totalEpisodeReward = self.totalEpisodeReward + reward
        return done
    def episode(self):
        state = self.env.reset()
        self.state = np.reshape(state, [self.numStateVariables,])
        self.totalEpisodeReward = 0
        done = False
        for stepNum in range(self.episodeSteps):
            done = self.goToNextState()
            if done:
                break
        if not done:
            self.goToNextState(endEarly=True)
        self.episodeRewards.append(self.totalEpisodeReward)
        fps = self.updateFps()
        print("REWARD: "+str(self.totalEpisodeReward)+" FPS: "+str(fps))
    def execute(self):
        for episode in range(self.maxEpisodes):
            self.episode()
            time.sleep(1)