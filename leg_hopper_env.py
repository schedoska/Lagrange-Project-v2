import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils.env_checker import check_env
from leg_hopper import LegHopperModel
from stable_baselines3 import PPO

class LegHopperEnv(gym.Env):

    def __init__(self, gravity: float, side_mass: float, 
               middle_mass: float, horizontal_bar_len: float, 
               time_step=float(0.001), render_mode=None):
        super(LegHopperEnv, self).__init__()
        self.render_mode = render_mode
        self.time_step = time_step

        self.action_space = spaces.Box(
            low=np.array([-100,-10]), 
            high=np.array([100,10]), 
            dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.ones(12) * -np.inf, 
            high=np.ones(12) * np.inf, 
            dtype=np.float64)
        
        self.robot = LegHopperModel(gravity, side_mass, 
                       middle_mass, horizontal_bar_len)
        print("Generating leg hopper equations...")
        self.robot.generate_equations()
        print("Generating leg hopper equations... done.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.robot.reset(np.array([0, 0, 0.6, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0]))
        return self.robot.state, {}
    
    def step(self, action):
        self.robot.advance(self.time_step, action)
        current_x = self.robot.state[0]
        current_y = self.robot.state[1]
        current_q1 = self.robot.state[2]
        current_q2 = self.robot.state[3]
        current_theta = self.robot.state[4]
        
        x_reward = np.abs(current_x - 0.3)**2
        y_reward = np.abs(current_y - 0)**2
        q1_reward = np.abs(current_q1 - 0.6)**2
        q2_reward = np.abs(current_q2 - np.pi/2)**2
        theta_reward = np.abs(current_theta - 0)**2
        reward = -(x_reward + y_reward + q1_reward + q2_reward + theta_reward) + 10

        truncated = False if self.robot.time < 3.0 else True
        terminated = True if current_theta < -np.pi/2 or current_theta > np.pi/2 else False 
        terminated = True if current_q2 < 0 or current_q2 > np.pi else terminated 
        terminated = True if current_q1 > 5 or current_q1 < 0 else terminated
        terminated = True if current_y < -0.65 else terminated

        reward = reward - 1000 if terminated else reward 
        info = {}

        #print(self.robot.state)

        return (
            self.robot.state,
            reward,
            terminated,
            truncated,
            info,
        )
        
    def render(self):
        if self.render_mode == "console":
            print(f"-- Current time: {self.robot.time:.2f} --")
            print(self.robot.state)

    def close(self):
        pass


if __name__ == "__main__":
    env = LegHopperEnv(9, 0.3, 0.2, 0.2, time_step=0.002, render_mode="console")
    obs, _ = env.reset(seed=5)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0008)
    model.learn(total_timesteps=int(8e5), progress_bar=True)
    model.save("leg_hopper_model")
    del model  

    quit()





    env.render()

    n_steps = 20
    for step in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(np.array([-10, 0]))
        done = terminated or truncated
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break
