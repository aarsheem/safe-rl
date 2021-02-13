import numpy as np

class Evaluate():
    def __init__(self, env, policy, maxSteps, feature=-1):
        self.env = env
        self.policy = policy
        self.maxSteps = maxSteps
        #helps in creating noisy states
        self.feature = feature

    def episode(self):
        self.env.reset()
        steps = 0
        states = []
        actions = []
        rewards = []
        while (not self.env.isEnd) and steps < self.maxSteps:
            normState = self.env.normState()
            if self.feature != -1:
                normState = [normState[self.feature]]
            action = self.policy.sampleAction(normState)
            _, reward, _ = self.env.step(action)
            steps += 1
            states.append(normState)
            actions.append(action)
            rewards.append(reward)
        return states, actions, rewards

    def theta(self, theta):
        self.policy.parameters = np.array(theta)
