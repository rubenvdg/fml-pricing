import numdifftools as nd
import numpy as np

from bnb.problem import Problem
from bnb.naive_optimizer import (
    get_lhs_vee,
    get_p_r,
    get_q,
    get_lipschitz_constant,
    _get_lipschitz_constant
)


def test_get_lipschitz_constant():
    m = 2
    n = 3
    a_range = (-4.0, 4.0)
    b_range = (0.001, 0.01)
    seed = 1
    problem = Problem(n, m, a_range, b_range, seed)
    p = np.asarray([2, 3, 4])
    radius = 0.5
    lipschitz_constant = get_lipschitz_constant(p, radius, problem)

    p_samples = np.random.uniform(p - radius, p + radius, size=(1000, len(p)))
    grad = nd.Gradient(problem.revenue)
    lipschitz_constant_estimate = np.max(np.asarray([
        np.abs(grad(p_sample)) for p_sample in p_samples
    ]))

    assert lipschitz_constant > lipschitz_constant_estimate


def test__get_lipschitz_constant():
    
    lhs_vee = np.asarray([
        [0, 0, 1],
        [0, 0, 0]
    ])
    rhs_vee = - lhs_vee
    q_ub_r = np.asarray([
        [0.1, 0.5, 0.2],
        [0.1, 0.4, 0.9]
    ])
    b = np.asarray([0, 0, 1])
    w = np.asarray([0.2, 0.8])
    L = _get_lipschitz_constant(lhs_vee, rhs_vee, q_ub_r, b, w)
    expectedL = w[0] * q_ub_r[0, 2]
    np.testing.assert_allclose(L, expectedL)


def test_get_q():
    
    A = np.asarray([[1, 2], [0.5, 0.1], [3, 3]])
    b = np.asarray([4, 2])
    p = np.asarray([[1, 5], [0.1, 2]])
    
    q = get_q(p, A, b)
    
    for i, c in zip(range(2), range(3)):
        utility = A[c] - p[i] * b
        exp_utility = np.exp(utility)
        q_expected = exp_utility / (1 + np.sum(exp_utility))
        np.testing.assert_allclose(q_expected[i], q[c, i])


def test_get_lhs_vee():
    
    b = np.asarray([1, 2, 3])
    p = np.asarray([0, 1, 2])
    q = np.asarray([
        [0.5, 0.6, 0.7],
        [0.1, 0.9, 0.2]
    ])
    
    lhs_vee = get_lhs_vee(q, p, b)
    
    for i, c in zip(range(3), range(2)):
        qp = q[c, :] * p
        qp[i] = 0
        expected = np.sum(qp) + 1 / b[i]
        np.testing.assert_allclose(lhs_vee[c, i], expected)


def test_get_p_r():

    p = np.asarray([1, 2])
    radius = 0.1
    
    p_lb_r, p_ub_r = get_p_r(p, radius)
    
    p_ub_r_expected = np.asarray([
        [0.9, 2.1],
        [1.1, 1.9]
    ])
    
    p_lb_r_expected = np.asarray([
        [1.1, 1.9],
        [0.9, 2.1]
    ])
    
    np.testing.assert_array_equal(p_ub_r, p_ub_r_expected)
    np.testing.assert_array_equal(p_lb_r, p_lb_r_expected)
