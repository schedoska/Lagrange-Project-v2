import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env

class ArrayCrawlerEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10, render_mode="console"):
        super(ArrayCrawlerEnv, self).__init__()
        self.render_mode = render_mode

        self.grid_size = grid_size
        self.agent_location = 0
        self.target_location = 0

        # Define action and observation space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = gym.spaces.Box(
            np.array([0, 0]), np.array([grid_size, grid_size]), dtype=np.int32)
        
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.target_location = self.agent_location = self.np_random.integers(0, self.grid_size)
        while self.target_location == self.agent_location:
            self.target_location = self.np_random.integers(0, self.grid_size)

        return np.array([self.agent_location, self.target_location], dtype=np.int32), {}  # empty info dict

    def step(self, action):
        if action == self.LEFT:
            self.agent_location -= 1
        elif action == self.RIGHT:
            self.agent_location += 1

        self.agent_location = np.clip(self.agent_location, 0, self.grid_size)

        # Are we at the left of the grid?
        terminated = bool(self.target_location == self.agent_location)
        truncated = False 

        reward = 1 if self.agent_location == self.target_location else -0.1

        info = {}

        return (
            np.array([self.agent_location, self.target_location], dtype=np.int32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            g = ["."] * self.grid_size
            g[self.agent_location] = "X"
            g[self.target_location] = "T"
            print(g)

    def close(self):
        pass




env = ArrayCrawlerEnv(grid_size=10, render_mode="console")
obs, _ = env.reset(seed=66)
env.render()

n_steps = 20
for step in range(n_steps):
    obs, reward, terminated, truncated, info = env.step(0)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break