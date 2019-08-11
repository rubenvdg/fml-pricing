import numpy as np
from bnb.naive_optimizer import get_lhs_vee, get_p_r, get_q


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
