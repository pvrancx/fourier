import numpy as np
import copy

class LinearOptionsAgent(object):
    '''Agent that applies fixed set of linear options - no learning'''
    def __init__(self, proj, options, betas, inits):
        self.n_options = len(options)
        self.options= options
        self.beta_fn = betas
        self.init_fn = inits
        self.option = None
        self.option_idx = 0
        self.term = False
        self.proj = proj
        self.cfg= {}



    def select_option(self, phi, opt_n):
        # select random option
        idx = self.random.choice(opt_n)
        return (idx, self.options[idx])

    def get_config(self):
        return self.cfg

    def select_action(self, obs):
        phi = self.proj.phi(obs)
        opts = self.get_valid_options(obs)
        if self.option is None:
            self.option_idx, self.option = self.select_option(phi, opts)
        return self.option(obs)

    def start(self, obs):
        opt_n = self.get_valid_options(obs)
        self.phi = self.proj.phi(obs)
        self.option_idx, self.option = self.select_option(self.phi, opt_n)
        self.act = self.option(obs)
        self.opts = self.get_matching_options(obs,self.act)
        return self.act

    def get_matching_options(self, obs, a):
        '''return indices of options that  select a given phi'''
        mask = np.array([o(obs)==a for o in self.options])
        return np.arange(self.n_options)[mask]

    def get_valid_options(self, obs):
        '''indices of options that can be triggered in obs'''
        mask = np.array([init(obs) for init in self.init_fn])
        return np.arange(self.n_options)[mask]

    def get_option_betas(self, obs):
        '''get termination probs for options given obs'''
        return np.array([beta(obs) for beta in self.beta_fn])

    def do_learning(self, rew, phi_n, done, beta_n):
        #put smart stuff here
        pass

    def update(self, obs_n, rew, done):
        # feature representation o fnext state
        phi_n = self.proj.phi(obs_n)
        # next state termination probs
        beta_n = self.get_option_betas(obs_n)
        # valid options in this state:
        opt_n = self.get_valid_options(obs_n)
        # check for termination
        self.term = beta_n[self.option_idx] > np.random.rand()
        # learning
        self.do_learning(rew, phi_n, done, beta_n, opt_n)
        # select new option if current one terminated
        if self.term:
            #activate new option
            self.option_idx, self.option = self.select_option(phi_n, opt_n)
            self.term = False
        # select action according to current option
        self.act = self.option(obs_n)
        self.phi = phi_n
        self.opts = self.get_matching_options(obs_n,self.act)
        return self.act

class OptionController(LinearOptionsAgent):
    ''' Base class for learning control with linear options'''
    def __init__(self, gamma=.99, epsilon=.01, beta_eps =0., **kwargs):
        super(OptionController, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta_eps = beta_eps # beta bias
        self.theta = np.zeros((self.proj.n_features,self.n_options))
        self.cfg.update({'gamma':gamma,'epsilon':epsilon,'beta_eps':beta_eps})




    def select_option(self, phi, opt_n):
        ''' e-greedy option selection'''
        if np.random.rand() < self.epsilon:
            idx = np.random.choice(opt_n)
        else:
            vals = np.dot(self.theta[:,opt_n].T, phi)
            max_idx = opt_n[vals == np.max(vals)]
            idx = np.random.choice(max_idx)
        return (idx, self.options[idx])


class SMDPQLearner(OptionController):
    '''Model free linear semimdp option learner'''
    def __init__(self, alpha, **kwargs):
        super(SMDPQLearner,self).__init__(**kwargs)
        self.alpha= alpha
        self.ro = 0.
        self.gamma_k = 1.
        self.option_phi = None
        self.beta_prod = 1.
        self.cfg.update({'alpha':alpha})

    def start(self, obs):
        super(SMDPQLearner, self).start(obs)
        self.option_phi = copy.copy(self.phi) # log starting phi
        self.ro = 0. # start accumulating option reward
        self.gamma_k = 1. #reset option discount
        return self.act

    def do_learning(self, rew, phi_n, done, beta_n, opt_n):
        beta = beta_n[self.option_idx] # termination prob of current option
        if self.term:
            # option terminated, bootstrap
            delta = self.ro - np.dot(self.theta[:,self.option_idx], self.option_phi)
            if not done:
                vals_n = np.dot(self.theta[:,opt_n].T,phi_n)
                delta += self.gamma *self.gamma_k*(1. -self.beta_eps)* beta * np.max(vals_n)
            self.theta[:,self.option_idx] += self.alpha*delta*self.option_phi
            self.ro = 0.
            self.gamma_k = 1.
            self.option_phi = phi_n
            self.beta_prod = 1.
        else:
            # option continues, accumulate reward
            self.ro += self.gamma_k * rew
            self.gamma_k *= self.gamma
            self.beta_prod *= (1.-beta)

class IntraOptionQLearner(OptionController):
    '''Model free linear intra option learner'''
    def __init__(self, alpha, **kwargs):
        super(IntraOptionQLearner,self).__init__(**kwargs)
        self.alpha= alpha
        self.cfg.update({'alpha':alpha})


    def do_learning(self, rew, phi_n, done, beta_n, opt_n):
        self.intra_option_value_learning(rew,phi_n,done,beta_n, opt_n)

    def intra_option_value_learning(self,rew,phi_n,done,beta_n, opt_n):
        '''update option Q-vals using TD rule'''
        phi = self.phi
        act = self.act

        vals = np.dot(self.theta.T,phi) # start state vals
        vals_n = np.dot(self.theta.T,phi_n) # next state vals_n
        max_val_n = np.max(vals[opt_n])
        #intra option value learning
        for idx in self.opts:
            delta = rew-vals[idx]
            if not done:
                U = (1.- beta_n[idx])*vals_n[idx] + (1.-self.beta_eps)*beta_n[idx]*max_val_n
                delta += self.gamma*U
            self.theta[:,idx] += self.alpha*delta*phi

class IntraOptionPlanner(OptionController):
    '''Planning based linear option learner'''

    def __init__(self, alpha1, alpha2, **kwargs):
        super(IntraOptionPlanner,self).__init__(**kwargs)

        self.alpha1 = alpha1 # model learning rate
        self.alpha2 = alpha2 # option value learning rate

        n_feat = self.proj.n_features
        # termination models
        self.F = np.zeros((n_feat, n_feat, self.n_options))
        #reward models
        self.b = np.zeros((n_feat,self.n_options))
        self.cfg.update({'alpha1':alpha1,'alpha2':alpha2})

    def do_learning(self, rew, phi_n, done, beta_n, opt_n):
        # update models
        self.intra_option_model_learning(rew,phi_n,done, beta_n)
        if self.term: #current option just terminated
            # update option values
            self.linear_option_planning(phi_n,opt_n)

    def intra_option_model_learning(self,rew,phi_n,done, beta_n):
        '''Learn the option models'''
        phi = self.phi
        act = self.act
        #intra option model learning
        for idx in self.opts:
            beta = beta_n[idx]
            # update termination model
            neta = self.phi - self.gamma * (1.-beta) * phi_n
            deltaF = self.gamma*beta*phi_n - np.dot(self.F[:,:,idx],neta)
            self.F[:,:,idx] += self.alpha1*np.outer(deltaF,neta)
            # update reward model
            delta = rew - np.dot(self.b[:,idx].T,phi)
            if not done:
                val_n = np.dot(self.b[:,idx].T, phi_n)
                delta += (1.-beta)*self.gamma*val_n
            self.b[:,idx] += self.alpha1 * delta * phi

    def linear_option_planning(self, phi, opt):
        '''Planning update for option Q-vals'''
        for o_idx in opt:
            # expected termination state for option
            phi_n = np.dot(self.F[:,:,o_idx],phi)
            # option values for expected termination state
            vals_n = np.dot(self.theta.T,phi_n) # restrict to valid options???
            # value of expected next state
            max_val_n = np.max(vals_n)
            #planning update
            rew = np.dot(self.b[:,o_idx].T,phi) #expected reward
            delta =  rew - np.dot(self.theta[:,o_idx].T,phi)
            delta += max_val_n
            self.theta[:,o_idx] += self.alpha2 * delta * phi
