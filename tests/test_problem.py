import warnings
import numpy as np
import numdifftools as nd
from bnb.problem import Problem


def test_revenue():
    np.random.seed(1)
    m = 2
    n = 1
    a_range = (-4.0, 4.0)
    b_range = (0.001, 0.01)
    seed = 1
    problem = Problem(n, m, a_range, b_range, seed)
    p = np.random.uniform(0, 1, size=n)
    a0, b0 = problem.segments[0].a, problem.segments[0].b
    a1, b1 = problem.segments[1].a, problem.segments[1].b
    rev0 = problem.w[0] * p * np.exp(a0 - b0 * p) / (1 + np.exp(a0 - b0 * p))
    rev1 = problem.w[1] * p * np.exp(a1 - b1 * p) / (1 + np.exp(a1 - b1 * p))
    assert np.allclose(problem.revenue(p), rev0 + rev1)


def test_gradient():
    np.random.seed(1)
    m = 2
    n = 2
    a_range = (-4.0, 4.0)
    b_range = (0.001, 0.01)
    seed = 1
    problem = Problem(n, m, a_range, b_range, seed)
    p = np.random.uniform(0, 1, size=n)
    der = nd.Gradient(problem.revenue)
    assert np.allclose(problem.gradient(p), der(p))
