import numpy as np
from scipy.optimize import root


class Problem:
    """Represents an instance of the FML pricing problem"""

    def __init__(self, n, m, a_range, b_range, seed, decision_variable='p'):
        self.seed, self.n, self.m = seed, n, m
        if self.seed:
            np.random.seed(self.seed)

        # sample random parameters
        self.w = np.random.uniform(0, 1, size=m)
        self.w /= np.sum(self.w)
        self.b = np.random.uniform(*b_range, size=n)
        a = [np.random.uniform(*a_range, size=n) for _ in range(self.m)]

        # initialize segments
        self.segments = [Segment(_a, self.b, _w) for _a, _w in zip(a, self.w)]
        self.A = np.asarray([segment.a for segment in self.segments])
        self.B = np.asarray([segment.b for segment in self.segments])

        # compute price bounds 
        self.p_lb, self.p_ub = self.compute_price_bounds()

        # compute bounds on no-purchase probabilities
        for segment in self.segments:
            segment.x_lb = segment.no_purchase_probability(self.p_lb)
            segment.x_ub = segment.no_purchase_probability(self.p_ub)
        self.x_lb = np.asarray([segment.x_lb for segment in self.segments])
        self.x_ub = np.asarray([segment.x_ub for segment in self.segments])

        # define helper variables
        self.E = np.asarray([np.exp(segment.a) for segment in self.segments]).T
        self.k = self.E * self.w.reshape(1, -1) / self.b.reshape(-1, 1)

    
    def compute_price_bounds(self):
        # compute upper bound on segment revenues
        ub_pi = []
        for segment in self.segments:
            res = root(self.pi_c, 5.0, args=(segment, self.b))
            if not res.success:
                raise Exception('Root finding failed.')
            else:
                ub_pi.append(res.x)

        return 1 / self.b, 1 / self.b + np.max(ub_pi)

    @staticmethod
    def pi_c(pi, segment, b):
        return pi - np.sum(1 / b * np.exp(segment.a - 1 - b * pi))


class Segment:
    ''' Represents a customer segment  '''

    def __init__(self, a, b, w):
        self.a = a
        self.b = b
        self.w = w
        self.x_lb = None
        self.x_ub = None

    def no_purchase_probability(self, p):
        return 1 / (1 + np.sum(np.exp(self.a - self.b * p)))
