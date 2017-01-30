from sarsa import Sarsa
import numpy as np

def in_rect(p,rect):
    '''check if point lies within rectangle '''
    x1,y1,x2,y2 = rect
    return (p[0] < x2 and
            p[0] > x1 and
            p[1] < y2 and
            p[1] > y1)

def intersects(point,obj):
    '''check if point is within (bounding box of) object '''
    bb = (obj.min_x,obj.min_y,obj.max_x,obj.max_y)# bounding box
    return in_rect(point, bb)

def is_valid(p, pb_env):
    ''' check that point doesn't interesct with pinball obstacles'''
    for obs in pb_env.environment.obstacles:
        if intersects(p, obs):
            return False
    return True

def rect2points(rect):
    '''give rect as list of corner points'''
    x1,y1,x2,y2 = rect
    return [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]

def random_point(rect, pb_env):
    '''select random valid point within rectangular region'''
    x1,y1,x2,y2 = rect
    p = np.array((x1,x2))+np.random.rand(2)*(x2-x1,y2-y1)
    while not is_valid(p, pb_env):
        # will run forever if no valid points can be found
        p = np.array((x1,x2))+np.random.rand(2)*(x2-x1,y2-y1)
    return p

def set_state(pb_env, state):
    '''overwrite state of pinball env (ugly...)'''
    pb_env.reset()
    pb_env.environment.ball.position[0] = state[0]
    pb_env.environment.ball.position[1] = state[1]
    pb_env.environment.ball.xdot = state[2]
    pb_env.environment.ball.ydot = state[3]
    pb_env.state = state
    return pb_env._get_ob()

def reset_pb(pb_env, region):
    '''reset pinball env to random initial state within region'''
    p = random_point(region,pb_env)
    v_low = pb_env.observation_space.low[-2:]
    v_high = pb_env.observation_space.high[-2:]
    v = v_low + np.random.rand(2) *(v_high-v_low)
    set_state(pb_env, np.array((p[0],p[1],v[0],v[1])))
    return pb_env._get_ob()

def load_option(save_file, eps=0.):
    import cPickle as pickle
    with open(save_file, 'rb') as f:
        agent = pickle.load(f)
    agent.epsilon = eps
    return lambda o: agent.select_action(o)[0]


def learn_option(init_region, target_region, pb_env, proj,
                 alpha = .001,
                 epsilon = .01,
                 gamma = .99,
                 n_eps=100):
    agent = Sarsa(proj,pb_env.action_space.n,
                  alpha=alpha, epsilon=epsilon, gamma=gamma)
    for e in range(n_eps):
        print e
        obs = reset_pb(pb_env, init_region)
        done = False
        act = agent.start(obs)
        while not done:
            obs_n, _, finish, _ =  pb_env.step(act)
            if finish:
                obs_n = reset_pb(pb_env, init_region)
            done = in_rect(obs_n[:2],target_region)
            rew = float(done)*100.
            act = agent.update(obs_n, rew, done)
    return agent
