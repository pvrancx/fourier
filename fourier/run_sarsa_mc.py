import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sarsa import Sarsa
from fourier import FourierBasis
import gym





n_eps = 1000
alpha = .001
epsilon = .0
gamma = 1.
order = 5

env = gym.make('MountainCar-v0')
n_actions = env.action_space.n

low, high = env.observation_space.low, env.observation_space.high
proj = FourierBasis(low, high, order=order)
agent = Sarsa(proj,n_actions,alpha=alpha, epsilon=epsilon, gamma=gamma)

steps_log =[]
for e in range(n_eps):
    done = False
    obs = env.reset()
    act = agent.start(obs)
    steps = 0
    while not done:
        obs_n, rew, done, _ = env.step(act)
        act = agent.update(obs_n, rew, done)
        steps += 1
    print 'episode %d, steps: %d'%(e,steps)
    steps_log.append(steps)

#plot result
Xs, Ys = np.meshgrid(np.linspace(low[0],high[0],100),
                     np.linspace(low[1],high[1],100))
Qs = np.zeros( Xs.shape+(n_actions,))

for i in range(Xs.shape[0]):
    for j in range(Xs.shape[1]):
        s = np.array(Xs[i,j],Ys[i,j])
        phi = proj.phi(s)
        Qs[i,j,:] = np.dot(phi,agent.theta)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_surface(Xs, Ys, -np.max(Qs,axis=2))
ax.set_title('value function')

ax = fig.add_subplot(1, 3, 2)
ax.matshow(np.argmax(Qs,axis=-1))
ax.set_title('policy')

ax = fig.add_subplot(1, 3, 3)
ax.plot(steps_log)
ax.set_title('steps')

plt.show()
