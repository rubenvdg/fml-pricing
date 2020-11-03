import numpy as np
from cvxopt import matrix, solvers
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.special import xlogy  # pylint: disable-msg=E0611
import logging
from .branchandbound import BranchAndBound, Cube

solvers.options["show_progress"] = False

# CNST = 1e6  # for normalization of optimization problem
TOL = None  # relative tolerance for optimization algorithms
SOLVER_OPTIONS = {"maxiter": int(1e6)}
SAMPLE_RANGE = (-50, -20)  # for sampling starting values of the dual optimization problem


class FMLSolver(BranchAndBound):
    """
    Represents the branch-and-bound algorithm from the paper
    "Price Optimization Under the Finite-Mixture Logit Model" (2018)
    by R. van de Geer and A. V. den Boer
    available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3235432
    """

    def __init__(self, problem, *args, **kwargs):

        super().__init__((problem.x_lb, problem.x_ub), *args, **kwargs)

        self.problem = problem
        self.z_lb = np.exp(-problem.p_ub * problem.b)
        self.z_ub = np.exp(-problem.p_lb * problem.b)
        self.n = len(self.problem.p_lb)
        self.m = len(self.problem.x_lb)
        self.k = problem.S * problem.w.reshape(1, -1) / problem.b.reshape(-1, 1)
        self.lp_constraints = None
        self._init_lp()

    def compute_upper_bound(self, cube: Cube) -> float:

        if np.any(cube.center - cube.radius > self.problem.x_ub):
            # Cube completely outside X -> set upper bound low to disregard
            cube.objective_lb = 0.0
            return -np.inf

        self._solve_lp(cube)
        cube_feasible = True if self.lp["status"] == "optimal" else False

        if not cube_feasible:
            # Cube completely outside X -> set upper bound low to disregard
            cube.objective_lb = 0.0
            return -np.inf

        # Shift cube such that center is feasible
        x_delta, r_delta = self._get_x_r_delta(cube)

        # Compute bounds
        lipschitz_bound = self.compute_lipschitz_upper_bound(cube, x_delta, r_delta)
        alternative_bound = self.compute_alternative_bound(cube, x_delta, r_delta)

        return np.min([lipschitz_bound, alternative_bound])

    def compute_alternative_bound(self, cube, x_delta, r_delta):

        logger = logging.getLogger(__name__)
        opt, cnst = self._compute_alternative_bound(x_delta, r_delta, cube)
        ub = -opt.fun * cnst

        if not opt.success:
            logger.warning("Optimization failed (no convergence): %s.", opt.message)
            return np.inf

        return ub

    def _compute_alternative_bound(self, x, r, cube):

        n, m, problem = self.n, self.m, self.problem

        def obj(xi, cnst):
            xi_ = xi.reshape((n, m))
            obj_ = np.sum((xlogy(xi_, xi_ / (x + r)) * problem.S * problem.w).T / problem.b) / cnst
            jac_ = np.where(
                xi_ > 0,
                (((1 + np.log(xi_ / (x + r))) * problem.S * problem.w).T / problem.b).T / cnst,
                -np.ones((n, m)),
            ).reshape((n * m,))
            return obj_, jac_

        ub = (np.ones((n, m)) * np.exp(-1) * (x + r)).reshape((n * m,))
        bounds = [(0, ub_) for ub_ in ub]

        def cnstr(xi):
            xi_ = xi.reshape((n, m))
            S_xi = np.sum(problem.S * xi_, axis=0)
            # return np.hstack([S_xi - (1 - np.maximum(x - r, self.problem.x_lb)), 1 - x + r - S_xi])
            return np.hstack([S_xi - (1 - x - r), 1 - x + r - S_xi])

        cnstrs = [{"type": "ineq", "fun": cnstr}]
        xi_start = np.outer(cube.z_opt, x)
        # cnst = np.linalg.norm(obj(xi_start, 1.0)[1])
        cnst = 1e9

        with np.errstate(all="ignore"):  # suppress overflows during optimization
            opt = minimize(
                obj, xi_start, args=(cnst,), bounds=bounds, constraints=cnstrs, options=SOLVER_OPTIONS, jac=True
            )

        return opt, cnst

    def compute_lipschitz_upper_bound(self, cube, x_delta, r_delta):

        with np.errstate(all="ignore"):  # suppress overflows during optimization
            min_problem = self._minimize_dual_lipschitz_bound(cube, x_delta)

        cube.theta_start = min_problem.x
        cube.objective_lb = min_problem.fun

        with np.errstate(all="ignore"):
            dual_dx_norm_ub = self._dual_dx_norm_ub(min_problem.x, x_delta, r_delta)
        return cube.objective_lb + dual_dx_norm_ub * r_delta

    def compute_lower_bound(self, cube: Cube) -> float:
        return cube.objective_lb

    def _get_x_r_delta(self, cube):
        n, m = self.n, self.m
        delta_plus = np.asarray(self.lp["x"])[n : n + m, 0]
        delta_min = np.asarray(self.lp["x"])[n + m : n + 2 * m, 0]
        u = (1 - cube.center) / cube.center - delta_plus + delta_min
        x_delta = 1 / (1 + u)
        r_delta = cube.radius + norm(cube.center - x_delta, ord=np.inf)
        return x_delta, r_delta

    def _minimize_dual_lipschitz_bound(self, cube, x_delta, tries=1000):

        mu = (1 - x_delta) / x_delta
        kx = self.k @ x_delta
        m, n = self.m, self.n

        def _dual(theta):
            lam, nu_lb, nu_ub = theta[:m], np.exp(theta[m : n + m]), np.exp(theta[n + m :])
            z = np.exp((self.problem.S @ lam - nu_ub + nu_lb) / kx - 1)
            obj_ = kx @ z - lam @ mu - nu_lb @ self.z_lb + nu_ub @ self.z_ub
            jac_ = np.hstack([z @ self.problem.S - mu, (z - self.z_lb) * nu_lb, (self.z_ub - z) * nu_ub])
            return obj_, jac_

        theta_start = np.random.uniform(*SAMPLE_RANGE, size=2 * n + m) if cube.theta_start is None else cube.theta_start

        min_problem = None
        for _ in range(tries):
            min_problem = minimize(_dual, theta_start, jac=True, tol=TOL)
            if min_problem.success:
                theta = min_problem.x
                lam, nu_lb, nu_ub = theta[:m], np.exp(theta[m : n + m]), np.exp(theta[n + m :])
                cube.z_opt = np.exp((self.problem.S @ lam - nu_ub + nu_lb) / kx - 1)
                return min_problem
            theta_start = np.random.uniform(*SAMPLE_RANGE, size=2 * n + m)

        raise ValueError(
            f"Dual optimization failed at cube.center {1 / (1 + mu)} "
            f"and cube.radius {cube.radius}, opt: \n{min_problem}, x_lb: {self.problem.x_lb}."
        )

    def _solve_lp(self, cube):

        x = cube.center
        u = (1 - x) / x
        b_eq = matrix(u)
        r = cube.radius

        b_ub_6 = u - (1 - (x + r)) / (x + r)
        b_ub_7 = (1 - (x - r)) / (x - r) - u
        cnstr = self.lp_constraints
        b_ub = matrix([cnstr["b_ub"], matrix(np.hstack([b_ub_6, b_ub_7]))])

        self.lp = solvers.lp(
            cnstr["c"],
            cnstr["A_ub"],
            b_ub,
            cnstr["A_eq"],
            b_eq,
            solver="glpk",
            options={"glpk": {"msg_lev": "GLP_MSG_OFF"}},
        )

    def _dual_dx_norm_ub(self, theta, x, r):
        """Compute L(x,r) defined in the accompanying paper. """
        n, m = self.n, self.m
        lam, nu_lb, nu_ub = theta[:m], np.exp(theta[m : n + m]), np.exp(theta[m + n :])

        log_z = (self.problem.S @ lam - nu_ub + nu_lb) / (self.k @ (x + r)) - 1
        lhs = (
            np.divide(
                lam,
                np.where(
                    lam > 0,
                    np.square(np.maximum(self.problem.x_lb, x - r)),
                    np.square(np.minimum(self.problem.x_ub, x + r)),
                ),
            )
            - self.problem.w * ((np.exp(log_z) * log_z / self.problem.b) @ self.problem.S)
        )

        log_z = (self.problem.S @ lam - nu_ub + nu_lb) / (self.k @ np.maximum(self.problem.x_lb, x - r)) - 1
        rhs = (
            np.divide(
                -lam,
                np.where(
                    lam > 0,
                    np.square(np.minimum(self.problem.x_ub, x + r)),
                    np.square(np.maximum(self.problem.x_lb, x - r)),
                ),
            )
            + self.problem.w * ((np.exp(log_z) * log_z / self.problem.b) @ self.problem.S)
        )

        return np.max(np.hstack((lhs, rhs)))

    def _init_lp(self):
        """Initialize the linear program for checking feasibility.

        Decision variables are respectively:
            - z, length: n
            - delta^+, lenth: m
            - delta^-, length: m
            - gamma, length: 1
        """

        n, m = self.n, self.m

        c = matrix(np.hstack([np.zeros(n + 2 * m), np.ones(1)]))
        A_eq = matrix(np.hstack([self.problem.S.T, np.identity(m), -np.identity(m), np.zeros((m, 1))]))

        b_ub_1 = np.zeros(m)
        b_ub_2 = -self.z_lb
        b_ub_3 = np.zeros(m)
        b_ub_4 = self.z_ub
        b_ub_5 = np.zeros(m)
        b_ub = matrix(np.hstack([b_ub_1, b_ub_2, b_ub_3, b_ub_4, b_ub_5]))

        A_ub_1 = np.hstack([np.zeros((m, n)), np.identity(m), np.identity(m), -np.ones((m, 1))])
        A_ub_2 = np.hstack([-np.identity(n), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, 1))])
        A_ub_3 = np.hstack([np.zeros((m, n)), -np.identity(m), np.zeros((m, m)), np.zeros((m, 1))])
        A_ub_4 = np.hstack([np.identity(n), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, 1))])
        A_ub_5 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), -np.identity(m), np.zeros((m, 1))])
        A_ub_6 = np.hstack([np.zeros((m, n)), np.identity(m), np.zeros((m, m)), np.zeros((m, 1))])
        A_ub_7 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), np.identity(m), np.zeros((m, 1))])
        A_ub = matrix(np.vstack([A_ub_1, A_ub_2, A_ub_3, A_ub_4, A_ub_5, A_ub_6, A_ub_7]))

        self.lp_constraints = {"c": c, "A_ub": A_ub, "A_eq": A_eq, "b_ub": b_ub}
