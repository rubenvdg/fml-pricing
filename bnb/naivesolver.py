import numpy as np

def get_lipschitz_constant(p, radius, problem):
    A, b, w = problem.A, problem.b, problem.w
    p_lb_r, p_ub_r = get_p_r(p, radius)
    q_lb_r = get_q(p_lb_r, A, b)
    q_ub_r = get_q(p_ub_r, A, b)
    p_ub_local = p + radius
    rhs_vee = p_ub_local * (1 - q_lb_r)
    lhs_vee = get_lhs_vee(q_ub_r, p_ub_local, b)
    return _get_lipschitz_constant(lhs_vee, rhs_vee, q_ub_r, b, w)


def get_lhs_vee(q: np.ndarray, p: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the lefthandside of the vee operator
    
    Parameters
    ----------
    q : np.ndarray
        Array with probabilities of dimension (m, n). 
    p : np.ndarray
        Array with prices of dimension (n, ).
    b : np.ndarray
        Array with price sensititivies of dimension (n, ). 
    
    Returns
    -------
    np.ndarray
        Lefthandside of the vee operator, dimension (m, n)
    """
    p_q_ub = np.repeat(
        np.expand_dims(q * p, axis=1),
        len(p),
        axis=1
    )
    for p_q_ub_c in p_q_ub:
        np.fill_diagonal(p_q_ub_c, 0)
    return 1 / b + np.sum(p_q_ub, axis=2)


def get_p_r(p: np.ndarray, radius: float) -> np.ndarray:
    """Get price extremes
    
    Parameters
    ----------
    p : np.ndarray
        Price vector of dimension (n, 1)
    radius : float
        Float for constructing price extremes
    
    Returns
    -------
    np.ndarray
        Price matrix of dimension (n, n), the i'th row pertains to a price 
        vector where the i'th element is low and the rest is high. 
    """
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


def _get_lipschitz_constant(lhs_vee, rhs_vee, q_ub_r, b, w):
    return np.max(
        np.sum(
            np.expand_dims(w, axis=-1)
            * q_ub_r
            * np.maximum(rhs_vee, lhs_vee),
            axis=0
        )
        * b
    )
