from sarsa import Sarsa
from fourier import FourierBasis
import gym
import gym_pinball
import matplotlib.pyplot as plt
import numpy as np
import os
import cPickle as pickle

def running_average(x,w):
    return np.convolve(x,np.ones(w),mode='valid')/w

n_eps = 100
alpha = .001
epsilon = .01
gamma = 1.
order = 5
save_file = 'sarsa_chkpt.pkl'

env = gym.make('PinBall-v0')
n_actions = env.action_space.n

low, high = env.observation_space.low, env.observation_space.high
proj = FourierBasis(low, high, order=order)

if os.path.isfile(save_file):
    with open(save_file,'rb') as f:
        agent = pickle.load(f)
else:
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

# save learner
with open(save_file,'wb') as f:
    pickle.dump(agent, f)

# demo run
done = False
obs = env.reset()
while not done:
    env.render()
    obs, _, done, _ = env.step( agent.select_action(obs)[0] )

env.close()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(running_average(rew_log,10))
ax.set_title('return')

ax = fig.add_subplot(1, 2, 2)
ax.plot(running_average(step_log,10))
ax.set_title('steps')
plt.show()
