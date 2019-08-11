import numpy as np


def get_lhs_vee(q, p, b):
    p_q_ub = np.repeat(
        np.expand_dims(q * p, axis=1),
        len(p),
        axis=1
    )
    for p_q_ub_c in p_q_ub:
        np.fill_diagonal(p_q_ub_c, 0)
    return 1 / b + np.sum(p_q_ub, axis=2)


def get_p_r(p, radius):
    n = len(p)
    delta = np.full((n, n), radius)
    np.fill_diagonal(delta, -radius)
    return p - delta, p + delta


def get_q(p, A, b):
    """p is square, compute purchase prob of c for i if prices are p[i]"""
    m = A.shape[0]
    utilities = np.subtract(
        np.expand_dims(A, axis=1),
        np.repeat(np.expand_dims(p * b, axis=0), m, axis=0)
    )
    exp_utilities = np.exp(utilities)
    return np.divide(
        np.diagonal(exp_utilities, axis1=1, axis2=2),
        1 + np.sum(exp_utilities, axis=2)
    )


def get_lipschitz_constant(lhs_vee, rhs_vee, q_ub_r, b, w):
    return np.max(
        np.sum(
            np.expand_dims(w, axis=-1)
            * q_ub_r
            * np.maximum(rhs_vee, lhs_vee),
            axis=0
        )
        * b
    )
