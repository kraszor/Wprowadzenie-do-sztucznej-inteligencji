import numpy as np
import gym


class QLearning:
    def __init__(self, name, gamma, beta, eps) -> None:
        self.size = name
        self.env = gym.make(name, is_slippery=True)
        self.env.reset()
        self.Q = None
        self.p_exploration = eps
        self.gamma = gamma
        self.beta = beta
        self.success = 0
        self.rewards_per_iteration = []

    def init_Q(self):
        observations = self.env.observation_space.n
        actions = self.env.action_space.n
        self.Q = np.zeros((observations, actions))

    def train(self, iters, actions_per_iter):
        for i in range(iters):
            state = self.env.reset()[0]
            terminal = False
            total_reward = 0
            for _ in range(actions_per_iter):
                if np.random.uniform(0, 1) < self.p_exploration:
                    action = self.env.action_space.sample()
                else:
                    best = np.max(self.Q[state, :])
                    worst = np.min(self.Q[state, :])
                    if best != 0:
                        action = np.argmax(self.Q[state, :])
                    elif best == 0 and worst == 0:
                        action = self.env.action_space.sample()
                    elif best == 0 and worst < 0:
                        indexes = [0, 1, 2, 3]
                        arg_worst = np.argmin(self.Q[state, :])
                        indexes.pop(arg_worst)
                        action = np.random.choice(indexes)
                next_state, reward, terminal, truncated, debug = self.env.step(action)
                if reward == 1:
                    # reward = 100
                    self.success += 1
                # if reward == 0 and terminal:
                #     reward = -100
                delta = reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]
                self.Q[state, action] = self.Q[state, action] + self.beta * delta
                total_reward += reward
                if terminal:
                    break
                state = next_state
            self.rewards_per_iteration.append(total_reward)

    def test(self, iters, actions_per_iter):
        success = 0
        for i in range(iters):
            state = self.env.reset()[0]
            terminal = False
            for _ in range(actions_per_iter):
                action = np.argmax(self.Q[state, :])
                next_state, reward, terminal, truncated, debug = self.env.step(action)
                state = next_state
                if next_state == 63:
                    success += 1
                if terminal:
                    break
        self.success = success
