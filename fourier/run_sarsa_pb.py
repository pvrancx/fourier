from sarsa import Sarsa
from fourier import FourierBasis
import gym
import gym_pinball
import matplotlib.pyplot as plt
import numpy as np

def running_average(x,w):
    return np.convolve(x,np.ones(w),mode='valid')/w

n_eps = 100
alpha = .001
epsilon = .01
gamma = 1.
order = 5

env = gym.make('PinBall-v0')
n_actions = env.action_space.n

low, high = env.observation_space.low, env.observation_space.high
proj = FourierBasis(low, high, order=order)
agent = Sarsa(proj,n_actions,alpha=alpha, epsilon=epsilon, gamma=gamma)

rew_log = []
step_log = []
for e in range(n_eps):
    done = False
    obs = env.reset()
    act = agent.start(obs)
    steps = 0
    total_rew = 0.
    while not done:
        obs_n, rew, done, _ = env.step(act)
        act = agent.update(obs_n, rew, done)
        steps += 1
        total_rew += rew
    print 'episode %d -- steps: %d -- rew: %3.2f'%(e,steps,total_rew)
    rew_log.append(total_rew)
    step_log.append(steps)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(running_average(rew_log,10))
ax.set_title('return')

ax = fig.add_subplot(1, 2, 2)
ax.plot(running_average(step_log,10))
ax.set_title('steps')
plt.show()
