import numpy as np
import copy


class Sarsa(object):
    ''' Linear SARSA(lambda) with fourier basis '''
    def __init__(self, projector,n_actions, alpha=.005, gamma=.99, lambda_=.9,
        epsilon=.005):
        self.projector = projector
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.theta = np.zeros((projector.n_features, n_actions))

    def egreedy(self, values):
        '''egreedy action selection'''
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            max_v = np.max(values)
            max_idx = np.arange(self.n_actions)[values==max_v]
            return np.random.choice(max_idx)

    def select_action(self, obs):
        '''calculate current values and run egreedy action selection'''
        phi = self.projector.phi(obs)
        vals = np.dot(phi,self.theta)
        act = self.egreedy(vals)
        return act, phi, vals

    def start(self, obs):
        act, phi, vals = self.select_action(obs)
        self.trace = np.zeros_like(self.theta)
        self.action = act
        self.phi = phi
        return act

    def update(self, obs_n, rew, term):
        self.trace *= self.gamma*self.lambda_
        self.trace[:,self.action] += self.phi #accumulating trace
        self.trace = np.clip(self.trace,-5.,5.)
        vals = np.dot(self.phi,self.theta)
        delta = rew - vals[self.action]
        act_n = 0
        phi_n = 0.
        if not term:
            act_n, phi_n, vals_n = self.select_action(obs_n)
            delta += self.gamma * vals_n[act_n]
        self.theta += (self.alpha*
                       np.tile(self.projector.scaling, (self.n_actions,1)).T
                       * delta * self.trace)
        self.phi = phi_n
        self.action = act_n
        return act_n
