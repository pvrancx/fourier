import numpy as np


class FourierBasis(object):

    def __init__(self, low, high, order=3):
        assert np.all( low <= high), 'invalid bounds'

        self.n_dims = low.size
        self.low = low
        self.high = high
        self.order = order

        # based on rlpy implementation
        self.coeffs = np.indices((order,) * self.n_dims).reshape(
            (self.n_dims, -1)).T
        self.scaling = np.linalg.norm(self.coeffs, axis = 1)
        self.scaling[self.scaling==0.] = 1.
        self.scaling = 1./self.scaling
        self.n_features = self.coeffs.shape[0]

    def phi(self, obs):
        obs_norm = (obs-self.low) / (self.high-self.low)
        return np.cos(np.dot(self.coeffs*np.pi,obs_norm))
