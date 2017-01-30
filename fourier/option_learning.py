import gym
import gym_pinball
import pb_options as pb
import numpy as np
import cPickle as pickle
from fourier import FourierBasis
from gym.envs.classic_control import rendering

start_regions = [np.array([0.,0.,1.,1.])]
target_regions = [
                #np.array([.1,.8,.2,.9]),
                #np.array([.8,.8,.9,.9]),
                #np.array([.55,.65,.65,.75]),
                np.array([.25,.2,.35,.3]),
                np.array([.6,.2,.7,.3])]

n_eps = 10000
alpha = .001
epsilon = .01
gamma = .95
order = 5

for start_region in start_regions:
    for target_region in target_regions:
        save_file = '../options/option_'+str(start_region)+'_'+str(target_region)+'.pkl'

        env = gym.make('PinBall-v0')
        n_actions = env.action_space.n

        low, high = env.observation_space.low, env.observation_space.high
        proj = FourierBasis(low, high, order=order)

        ag = pb.learn_option(start_region, target_region, env, proj,
                            n_eps = n_eps,
                            alpha = alpha,
                            epsilon = epsilon,
                            gamma = gamma
                            )

        with open(save_file,'wb') as f:
            pickle.dump(ag,f)
