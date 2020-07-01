import numpy as np
from scipy.optimize import root


class OptimizationProblem:
    """Represents an instance of the FML pricing problem"""

    def __init__(self, a, b, w):

        #TODO: if python list, cast to np array
        #TODO: check dimensions
        self.n, self.m, self.b, self.w = len(b), len(a), b, w

        # initialize segments
        self.segments = [Segment(_a, b, _w) for _a, _w in zip(a, w)]
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
        self.k = self.E * w.reshape(1, -1) / b.reshape(-1, 1)


    def compute_price_bounds(self):
        # compute upper bound on segment revenues
        ub_pi = []
        self.segment_prices = []
        for segment in self.segments:
            res = root(self.pi_c, 5.0, args=(segment, self.b))
            if not res.success:
                raise Exception('Root finding failed.')
            else:
                ub_pi.append(res.x)
            segment.rev_opt = res.x
            segment.p_opt = 1 / self.b + res.x
            segment.x_opt = segment.no_purchase_probability(segment.p_opt)
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
        self.p_opt = None
        self.rev_opt = None

    def no_purchase_probability(self, p):
        return 1 - np.sum(self.purchase_probabilities(p))

    def purchase_probabilities(self, p):
        utilities = self.a - self.b * p
        return np.exp(utilities) / (1 + np.sum(np.exp(utilities)))

    def revenue(self, q):
        return np.sum(self.a / self.b * q) + np.sum(q / self.b) * np.log(1 - np.sum(q)) - np.sum(q / self.b * np.log(q))

