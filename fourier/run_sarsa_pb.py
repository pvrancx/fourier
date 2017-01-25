from sarsa import Sarsa
from fourier import FourierBasis
import gym
import gym_pinball


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
