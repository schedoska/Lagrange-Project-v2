import pickle
import scipy.integrate
from sympy.physics.mechanics import *
import sympy as sp
import numpy as np
from enum import Enum
from math import floor
from leg_hopper_env import LegHopperEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO


env = LegHopperEnv(9, 0.3, 0.2, 0.2, time_step=0.002, render_mode="console")
model = PPO.load("leg_hopper_model", env=env)

sim_time = 3.0
dt = 0.002
n_samples = floor(sim_time/dt)

t = 0
sola = np.zeros(shape=(n_samples, 12), dtype='float64')
sola_t = np.zeros(n_samples)

for k in range(1,n_samples):
    print(t)
    t = t + dt
    sola_t[k] = t
    action, _ = model.predict(env.robot.state, deterministic=True)
    state,reward,terminated,truncated,_ = env.step(action)
    print(action)
    sola[k,:] = state
        
# Quick plotting data
plt.plot(sola_t, sola[:,0])
plt.plot(sola_t, sola[:,1])
plt.show()

# Animation/Visualization
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(left=-0.7,right=0.7)
ax.set_ylim(top=1.5,bottom=-1)
ax.set_aspect('equal')

lp = 0.2

def animate(i):
    x0 = sola[i,0]
    y0 = sola[i,1]
    q1 = sola[i,2]
    q2 = sola[i,3]
    theta = sola[i,4]
        
    x1 = x0-np.cos(theta)*lp
    y1 = y0-np.sin(theta)*lp
    x2 = x0+np.cos(theta)*lp
    y2 = y0+np.sin(theta)*lp

    x3 = x0 - np.cos(q2+theta)*q1
    y3 = y0 - np.sin(q2+theta)*q1

    line.set_data(np.array([[x1,x2],[y1,y2]]))
    line2.set_data(np.array([[x0, x3],[y0, y3]]))
    return (line,line2)

ani = animation.FuncAnimation(
    fig, animate, len(sola[:,0]), interval=10, blit=False)
plt.show()