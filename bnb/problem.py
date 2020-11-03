import numpy as np
from scipy.optimize import root

np.set_printoptions(precision=4, suppress=True)


class OptimizationProblem:
    """Represents an instance of the FML pricing problem"""

    def __init__(self, a, b, w):

        self.n, self.m, self.b, self.w = len(b), len(a), b, w

        # Initialize segments
        self.segments = [Segment(_a, b, _w) for _a, _w in zip(a, w)]
        self.A = np.asarray([segment.a for segment in self.segments])  # n * m
        self.S = np.exp(self.A.T)  # m * n
        self.B = np.asarray([segment.b for segment in self.segments])

        # Compute price bounds
        self.p_lb, self.p_ub = self.compute_price_bounds()

        # Compute bounds on no-purchase probabilities
        for segment in self.segments:
            segment.x_lb = segment.no_purchase_probability(self.p_lb)
            segment.x_ub = segment.no_purchase_probability(self.p_ub)

        self.x_lb = np.asarray([segment.x_lb for segment in self.segments])
        self.x_ub = np.asarray([segment.x_ub for segment in self.segments])

    def compute_price_bounds(self):
        self._compute_optimal_revenue_per_segment()
        optimal_revenue_per_segment = [segment.rev_opt for segment in self.segments]
        return 1 / self.b, 1 / self.b + np.max(optimal_revenue_per_segment)

    def _compute_optimal_revenue_per_segment(self, starting_value=5.0):
        for segment in self.segments:
            result = root(self.pi_c, starting_value, args=(segment, self.b))
            rev_opt = result.x
            if not result.success:
                raise Exception("Root finding failed.")
            segment.rev_opt = rev_opt
            segment.p_opt = 1 / self.b + rev_opt
            segment.x_opt = segment.no_purchase_probability(segment.p_opt)

    @staticmethod
    def pi_c(pi, segment, b):
        return pi - np.sum(1 / b * np.exp(segment.a - 1 - b * pi))


class Segment:
    """ Represents a customer segment  """

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
