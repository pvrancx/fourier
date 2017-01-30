# fourier
Sarsa(Lambda) Reinforcement Learner with Fourier features for gym environments.

## Usage
See run_sarsa_mc.py or run_sarsa_pb.py for example usage.

## Results
Example basis functions over 2D state space:
![example basis functions](./img/bf.png)

Function approximation example:
![Function approximation example](./img/sup.png)

Results on MountainCar:
![MountainCar Results](./img/mc.png)

## Options

The repo includes a number of pre-learned option policies for the pinball domain. These can be loaded
from the options folder. Options are trained to move the ball from a starting
region to a goal region. The options are named  'option\_[starting region]\_[goal region]'.
Regions are defined as rectangles (xmin,ymin,xmax,ymax), i.e. opposite corners
of the rectangle. A region [0 0 1 1] means option is trained from any starting
position.

### Loading options
Option policies can be loaded using the load_option function from pb_options

> import pb_options as pb

> option_file = '../options/option\_\[ 0.  0.  1.  1.\]\_\[ 0.8  0.8  0.9  0.9\].pkl'

> option = pb.load_option(f,0.) # second parameter makes policy egreedy

Loaded option policies map observations to actions:

> import gym

> import gym_pinball

> obs = env.reset()

> act = option(obs)

See option_demo.py for an example.
