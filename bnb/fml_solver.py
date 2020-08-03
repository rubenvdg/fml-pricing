import numpy as np
from cvxopt import matrix, solvers
from jax import grad
from jax import numpy as jnp
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.special import xlogy  # pylint: disable-msg=E0611
import logging
from .branchandbound import BranchAndBound, Cube

solvers.options["show_progress"] = False


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
        # logger = logging.getLogger(__name__)

        if np.any(cube.center - cube.radius > self.problem.x_ub):
            # completely outside [x_lb, x_ub] -> set upper bound low to disregard
            cube.objective_lb = 0.0
            return -np.inf

        self._solve_lp(cube)
        cube_feasible = True if self.lp["status"] == "optimal" else False

        if not cube_feasible:
            # completely outside X -> set upper bound low to disregard
            cube.objective_lb = 0.0
            return -np.inf

        lipschitz_bound = self.compute_lipschitz_upper_bound(cube)
        # good_bound = np.inf
        # print(lipschitz_bound, good_bound)
        # if lipschitz_bound < 5 * cube.objective_lb:
        # if True:
        return lipschitz_bound
        # else:
        #     good_bound = self.compute_good_upper_bound(cube)
        #     print(lipschitz_bound, good_bound)
        #     return np.min([lipschitz_bound, good_bound])
        #     relaxation_bound = self.compute_relaxation_upper_bound(cube)
        #     return np.min([lipschitz_bound, relaxation_bound])

    def compute_lipschitz_upper_bound(self, cube):
        # logger = logging.getLogger(__name__)
        # shift cube so that center is feasible
        x_delta, r_delta = self._get_x_r_delta(cube)

        with np.errstate(over="ignore"):  # suppress overflows during optimization
            min_problem = self._minimize_dual_lipschitz_bound(cube, x_delta)

        cube.theta_start = min_problem.x
        cube.objective_lb = min_problem.fun

        # if cube not completely in positive orthant, we cannot bound it
        # if np.any(x_delta - r_delta < 0):
        if False:
            # print("x, r, x_delta, r_delta: ", cube.center, cube.radius, x_delta, r_delta)
            return np.inf
        else:
            dual_dx_norm_ub = self._dual_dx_norm_ub(min_problem.x, x_delta, r_delta)
            # if np.isinf(dual_dx_norm_ub):
                # print("ub: ", dual_dx_norm_ub)
            return cube.objective_lb + dual_dx_norm_ub * r_delta

    # def compute_good_upper_bound(self, cube):

    #     problem = self.problem
    #     n, m = self.n, self.m

    #     x_ub = cube.center + cube.radius
    #     x_lb = cube.center - cube.radius

    #     z_lb = np.exp(-problem.p_ub * problem.b)
    #     z_ub = np.exp(-problem.p_lb * problem.b)

    #     min_z = np.min(z_lb)
    #     z_lb /= min_z
    #     z_ub /= min_z

    #     S = np.exp(problem.A)

    #     def cnstr(theta, c):
    #         x, z = theta[:m], theta[m:] * min_z
    #         return 1 - x[c] - x[c] * (S[c] @ z)  # >= 0

    #     def objective(theta):
    #         x, z = theta[:m], theta[m:] * min_z
    #         return np.sum(xlogy(z, z) * (self.k @ x))

    #     def jac(theta):
    #         x, z = theta[:m], theta[m:] * min_z
    #         return np.hstack((
    #             xlogy(z, z) @ self.k,
    #             min_z * (1 + np.log(z)) * (self.k @ x)
    #         ))

    #     constraints = (
    #         [{"type": "ineq", "fun": lambda theta, c=c: cnstr(theta, c)} for c in range(m)]
    #     )

    #     theta_start = np.hstack((x_lb, z_lb))
    #     bounds = list(zip(x_lb, x_ub)) + list(zip(z_lb, z_ub))

    #     # x_delta, _ = self._get_x_r_delta(cube)
    #     # theta_start = np.hstack((x_lb, z_lb))
    #     # bounds = list(zip(x_lb, x_ub)) + [(z), None)] * n

    #     with np.errstate(all="ignore"):
    #         opt = minimize(
    #             objective,
    #             theta_start,
    #             jac=jac,
    #             bounds=bounds,
    #             constraints=constraints,
    #             options={"maxiter": 1e6},
    #         )

    #     if opt.success:
    #         return - opt.fun
    #     else:
    #         return np.inf

    # def compute_relaxation_upper_bound(self, cube):

    #     n, m = self.n, self.m
    #     x_lb, x_ub = cube.center - cube.radius, cube.center + cube.radius
    #     kx = self.k @ x_ub

    #     def _dual(theta):
    #         lam1 = np.exp(theta[:m])
    #         lam2 = np.exp(theta[m : 2 * m])
    #         lam3 = np.exp(theta[2 * m : 2 * m + n])
    #         lam4 = np.exp(theta[2 * m + n :])
    #         z = np.exp((self.problem.S @ (lam2 - lam1) - lam3 + lam4) / kx - 1)
    #         return (
    #             z @ self.k @ x_ub - lam1 @ (1 - 1 / x_lb) - lam2 @ (1 / x_ub - 1) + lam3 @ self.z_ub - lam4 @ self.z_lb
    #         )

    #     def _dual_gradient(theta):
    #         lam1 = np.exp(theta[:m])
    #         lam2 = np.exp(theta[m : 2 * m])
    #         lam3 = np.exp(theta[2 * m : 2 * m + n])
    #         lam4 = np.exp(theta[2 * m + n :])
    #         z = np.exp((self.problem.S @ (lam2 - lam1) - lam3 + lam4) / kx - 1)
    #         return np.hstack(
    #             [
    #                 (-z @ self.problem.S - (1 - 1 / x_lb)) * lam1,
    #                 (z @ self.problem.S - (1 / x_ub - 1)) * lam2,
    #                 (self.z_ub - z) * lam3,
    #                 (z - self.z_lb) * lam4,
    #             ]
    #         )

    #     theta_start = np.random.uniform(-50, -20, size=2 * m + 2 * n)
    #     with np.errstate(over="ignore"):  # suppress overflows during optimization
    #         min_problem = minimize(_dual, theta_start, jac=_dual_gradient)
    #     if min_problem.success:
    #         return min_problem.fun
    #     else:
    #         return np.inf

    def compute_lower_bound(self, cube: Cube) -> float:
        return cube.objective_lb

    def _get_x_r_delta(self, cube):
        n, m = self.n, self.m
        delta_plus = np.asarray(self.lp["x"])[n : n + m, 0]
        delta_min = np.asarray(self.lp["x"])[n + m : n + 2 * m, 0]
        u = (1 - cube.center) / cube.center - delta_plus + delta_min
        x_delta = 1 / (1 + u)
        r_delta = self.radius + norm(cube.center - x_delta, ord=np.inf)
        return x_delta, r_delta

    def _dual_dx_norm_ub(self, theta, x, r):
        """Compute L(x,r) defined in the accompanying paper. """
        n, m = self.n, self.m
        lam, nu_lb, nu_ub = theta[:m], np.exp(theta[m : n + m]), np.exp(theta[m + n :])

        log_z = (self.problem.S @ lam - nu_ub + nu_lb) / (self.k @ (x + r)) - 1
        lhs = np.divide(
            lam,
            np.where(
                lam > 0,
                np.square(np.maximum(self.problem.x_lb, x - r)),
                np.square(np.minimum(self.problem.x_ub, x + r)),
            ),
        ) - self.problem.w * ((np.exp(log_z) * log_z / self.problem.b) @ self.problem.S)

        log_z = (self.problem.S @ lam - nu_ub + nu_lb) / (self.k @ (x - r)) - 1
        rhs = np.divide(
            -lam,
            np.where(
                lam > 0,
                np.square(np.minimum(self.problem.x_ub, x + r)),
                np.square(np.maximum(self.problem.x_lb, x - r)),
            ),
        ) + self.problem.w * ((np.exp(log_z) * log_z / self.problem.b) @ self.problem.S)

        return np.max(np.hstack((lhs, rhs)))

    def _minimize_dual_lipschitz_bound(self, cube, x_delta, tries=200):

        mu = (1 - x_delta) / x_delta
        kx = self.k @ x_delta
        m, n = self.m, self.n

        def _dual(theta):
            lam, nu_lb, nu_ub = theta[:m], np.exp(theta[m : n + m]), np.exp(theta[n + m :])
            z = np.exp((self.problem.S @ lam - nu_ub + nu_lb) / kx - 1)
            return (
                kx @ z - lam @ mu - nu_lb @ self.z_lb + nu_ub @ self.z_ub,
                np.hstack([z @ self.problem.S - mu, (z - self.z_lb) * nu_lb, (self.z_ub - z) * nu_ub])
            )

        theta_start = np.random.uniform(-50, -20, size=2 * n + m) if cube.theta_start is None else cube.theta_start
        for _ in range(tries):
            min_problem = minimize(_dual, theta_start, jac=True)
            if min_problem.success:
                return min_problem
            theta_start = np.random.uniform(-50, -20, size=2 * n + m)

        raise Exception(
            f"Dual optimization failed at cube.center {1 / (1 + mu)} "
            f"and self.radius {self.radius}, opt: \n{min_problem}, x_lb: {self.problem.x_lb}."
        )

    def _solve_lp(self, cube):
        # logger = logging.getLogger(__name__)
        x = cube.center
        u = (1 - x) / x
        b_eq = matrix(u)
        r = self.radius

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

        # gamma >= delta^+ + delta^-
        A_ub_1 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), np.identity(m), -np.ones((m, 1))])
        # A_ub_1 = np.hstack([np.zeros((m, n)), np.identity(m), np.identity(m), -np.ones((m, 1))])
        A_ub_2 = np.hstack([-np.identity(n), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, 1))])
        A_ub_3 = np.hstack([np.zeros((m, n)), -np.identity(m), np.zeros((m, m)), np.zeros((m, 1))])
        A_ub_4 = np.hstack([np.identity(n), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, 1))])
        A_ub_5 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), -np.identity(m), np.zeros((m, 1))])
        A_ub_6 = np.hstack([np.zeros((m, n)), np.identity(m), np.zeros((m, m)), np.zeros((m, 1))])
        A_ub_7 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), np.identity(m), np.zeros((m, 1))])
        A_ub = matrix(np.vstack([A_ub_1, A_ub_2, A_ub_3, A_ub_4, A_ub_5, A_ub_6, A_ub_7]))

        self.lp_constraints = {"c": c, "A_ub": A_ub, "A_eq": A_eq, "b_ub": b_ub}
