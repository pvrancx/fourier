import gym
import gym_pinball
import pb_options as pb
import numpy as np

# options are named  option_[starting region]_[goal region]
# regions are given as rectangles (xmin,ymin,xmax,ymax)
# [0 0 1 1] means option is trained from any starting position
option_list = ['../options/option_[ 0.  0.  1.  1.]_[ 0.8  0.8  0.9  0.9].pkl', #upper right
           '../options/option_[ 0.  0.  1.  1.]_[ 0.55  0.65  0.65  0.75].pkl', #center
           '../options/option_[ 0.  0.  1.  1.]_[ 0.25  0.2   0.35  0.3 ].pkl', #lower left
           '../options/option_[ 0.  0.  1.  1.]_[ 0.6  0.2  0.7  0.3].pkl', #lower right
           '../options/option_[ 0.6   0.    1.    0.35]_goal.pkl' # finish

]

option_steps = 300

options = []
eps = 0.01 # can be used to make options egreedy

for f in option_list:
    options.append(pb.load_option(f,eps))

env = gym.make('PinBall-v0')

obs = env.reset()
done = False

#run each option in sequence for fixed amount of steps
for option in options:
    for _ in range(option_steps):
        if done:
            break
        env.render()
        act = option(obs)
        obs, rew, done, _ = env.step(act)
