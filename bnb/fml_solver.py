import numpy as np
from cvxopt import matrix, solvers
from jax import grad
from jax import numpy as jnp
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.special import xlogy  # pylint: disable-msg=E0611

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

        self.problem = problem
        bounds = (problem.x_lb, problem.x_ub)
        super().__init__(bounds, *args, **kwargs)

        self.z_lb = np.exp(-problem.p_ub * problem.b)
        self.z_ub = np.exp(-problem.p_lb * problem.b)
        self.n = len(self.problem.p_lb)
        self.m = len(self.problem.x_lb)

        self.lp_constraints = None
        self._init_lp()

    def compute_upper_bound(self, cube: Cube) -> float:

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

        # new = self.new_upper_bound_dual(cube)
        lipschitz = self.compute_lipschitz_upper_bound(cube)
        # print(lipschitz, new)
        # return np.min([lipschitz, new])
        return lipschitz

    def compute_lipschitz_upper_bound(self, cube):

        # shift cube so that center is feasible
        x_delta, r_delta = self._get_x_r_delta(cube)

        # if cube not completely in positive orthant, branch it
        if np.any(x_delta - r_delta < 0):
            cube.objective_lb = 0.0
            return np.inf  # set upper bound high, so that cube will be branched

        if cube.theta_start is None:
            cube.theta_start = np.random.uniform(size=2 * self.n + self.m)

        with np.errstate(all="ignore"):
            min_problem = self._minimize_dual(cube, x_delta)

        cube.theta_start = min_problem.x
        cube.objective_lb = min_problem.fun

        dual_dx_norm_ub = self._dual_dx_norm_ub(min_problem.x, x_delta, r_delta)

        # print(min_problem.fun, x_delta, r_delta, cube.objective_lb + dual_dx_norm_ub * r_delta)
        # theta = min_problem.x
        # m, n = self.m, self.n
        # lam, nu_lb, nu_ub = theta[:m], theta[m : n + m], theta[n + m :]
        # print("lam: ", lam)
        # print("nu_lb: ", nu_lb)
        # print("nu_ub: ", nu_ub)
        # print("nu_ub: ", nu_ub)
        # print("")
        return cube.objective_lb + dual_dx_norm_ub * r_delta

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
        """ Compute L(x,r) defined in the accompanying paper """
        n, m, E = self.n, self.m, self.problem.E
        x_lb, x_ub = self.problem.x_lb, self.problem.x_ub
        # x, r = cube.center, cube.radius

        lam, nu_lb, nu_ub = theta[:m], theta[m : n + m], theta[n + m :]

        # z = self.z_tilde(lam, nu_ub, nu_lb, self.problem.k @ x)
        # assert all(z >= self.z_lb) and all(z <= self.z_ub), (list(zip(z, self.z_lb)), list(zip(z, self.z_ub)))

        log_z = (E @ lam - nu_ub + nu_lb) / (self.problem.k @ (x + r)) - 1
        i = lam / (
            (lam > 0) * np.square(np.maximum(x_lb, x - r)) + (lam < 0) * np.square(np.minimum(x_ub, x + r))
        ) - self.problem.w * ((np.exp(log_z) * log_z / self.problem.b) @ E)

        log_z = (E @ lam - nu_ub + nu_lb) / (self.problem.k @ (x - r)) - 1
        ii = -lam / (
            (lam > 0) * np.square(np.minimum(x_ub, x + r)) + (lam < 0) * np.square(np.maximum(x_lb, x - r))
        ) + self.problem.w * ((np.exp(log_z) * log_z / self.problem.b) @ E)

        return np.max([np.max(i), np.max(ii)])

    def _minimize_dual(self, cube, x_delta, tries=10):

        mu = (1 - x_delta) / x_delta
        kx = self.problem.k @ x_delta
        theta_start = cube.theta_start

        for _ in range(tries):

            min_problem = minimize(
                self._dual,
                theta_start,
                args=(mu, kx),
                bounds=[(None, None)] * self.m + [(0, None)] * 2 * self.n,
                jac=self._dual_gradient,
                method="SLSQP",
                options={"maxiter": 1e5, "ftol": 1e-12},
            )

            if min_problem.success:
                return min_problem

            theta_start = np.random.uniform(size=2 * self.n + self.m)

        exc = (
            f"dual optimization failed at cube.center {1 / (1 + mu)}"
            f" and self.radius {self.radius}, opt: \n{min_problem}"
            f", x_lb: {self.problem.x_lb}."
        )

        raise Exception(exc)

    def _dual(self, theta, mu, kx):
        n, m = self.n, self.m
        lam, nu_lb, nu_ub = theta[:m], theta[m : n + m], theta[n + m :]
        z = self.z_tilde(lam, nu_ub, nu_lb, kx)
        return kx @ z - lam @ mu - nu_lb @ self.z_lb + nu_ub @ self.z_ub

    def z_tilde(self, lam, nu_ub, nu_lb, kx):
        return np.exp((self.problem.E @ lam - nu_ub + nu_lb) / kx - 1)

    def _dual_gradient(self, theta, mu, kx):
        n, m = self.n, self.m
        lam, nu_lb, nu_ub = theta[:m], theta[m : n + m], theta[n + m :]
        z = self.z_tilde(lam, nu_ub, nu_lb, kx)
        grad = np.hstack([z @ self.problem.E - mu, z - self.z_lb, self.z_ub - z])
        return grad

    def new_upper_bound_dual(self, cube):

        E, k = self.problem.E, self.problem.k
        n, m = self.n, self.m

        # x_delta, _ = self._get_x_r_delta(cube)

        x_ub = cube.center + cube.radius
        x_lb = cube.center - cube.radius
        # x_ub = x_delta
        # x_lb = x_delta

        z_lb = np.exp(-self.problem.p_ub * self.problem.b)
        z_ub = np.exp(-self.problem.p_lb * self.problem.b)

        kx_ub = k @ x_ub

        def _dual_new_ub(theta):
            lam1, lam2, lam3, lam4 = theta[:m], theta[m : 2 * m], theta[2 * m : 2 * m + n], theta[2 * m + n :]
            z = np.exp((E @ (lam2 - lam1) - lam3 + lam4) / kx_ub - 1)
            return z @ kx_ub - lam1 @ (1 - 1 / x_lb) - lam2 @ (1 / x_ub - 1) + lam3 @ z_ub - lam4 @ z_lb

        bounds = [(0, None)] * (2 * m + 2 * n)
        theta_start = np.random.uniform(size=2 * m + 2 * n)
        min_problem = minimize(
            _dual_new_ub, theta_start, bounds=bounds, method="SLSQP", options={"maxiter": 1e5, "ftol": 1e-12}
        )
        assert min_problem.success, min_problem
        # theta = min_problem.x
        # lam1, lam2, lam3, lam4 = theta[:m], theta[m : 2 * m], theta[2 * m : 2 * m + n], theta[2 * m + n :]
        # print("lam3: ", lam3)
        # print("lam4: ", lam4)
        return min_problem.fun

    def _solve_lp(self, cube):

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

        n, m, E = self.n, self.m, self.problem.E

        c = matrix(np.hstack([np.zeros(n + 2 * m), np.ones(1)]))
        A_eq = matrix(np.hstack([E.T, np.identity(m), -np.identity(m), np.zeros((m, 1))]))

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
