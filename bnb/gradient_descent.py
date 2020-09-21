import numpy as np
from scipy.optimize import minimize

from bnb.problem import OptimizationProblem


class GradientDescent(OptimizationProblem):
    def solve(self):

        max_ = -np.inf

        for _ in range(10):
            p_start = np.random.uniform(self.p_lb, self.p_ub)
            with np.errstate(all="ignore"):
                opt = minimize(
                    lambda p: -self.objective_function(p),
                    p_start,
                    options={"gtol": 1e-03},
                )
            if opt.success and -opt.fun > max_:
                max_ = -opt.fun
        return max_

    def objective_function(self, p):
        return np.sum(
            [
                segment.w * np.sum(p * segment.purchase_probabilities(p))
                for segment in self.segments
            ]
        )
