import numpy as np
def getColumn(arr, column):
    out = []
    for i in range(len(arr)):
        out.append(arr[i][column])
    return out
def oneHot(arr, maxLen):
    out = []
    for i in range(len(arr)):
        entry = np.zeros(maxLen)
        entry[arr[i]] = 1
        out.append(entry)
    return out
def assertShape(tensor, shape):
    np.testing.assert_array_equal(tensor.get_shape().as_list(), shape)
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
