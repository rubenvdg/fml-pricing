import numpy as np
from cvxopt import matrix, solvers
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.special import xlogy

from .branchandbound import BranchAndBound, Cube

solvers.options["show_progress"] = False

SCALE = 1e5


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
            # outside [x_lb, x_ub] -> set upper bound low as to not branch
            cube.objective_lb = 0.0
            return -np.inf

        self._solve_lp(cube)
        cube_feasible = True if self.lp["status"] == "optimal" else False

        if not cube_feasible:
            # set upper bound low, so that cube is not branched
            cube.objective_lb = 0.0
            return -np.inf

        lipschitz_upper_bound = self.compute_lipschitz_upper_bound(cube)
        if np.min(cube.center - cube.radius) < 0.10:
            new_ub = self.new_upper_bound(cube)
        else:
            new_ub = np.inf

        return np.min([lipschitz_upper_bound, new_ub])

    def new_upper_bound(self, cube):

        S = np.exp(self.problem.A)

        z_lb = np.exp(-self.problem.p_ub * self.problem.b)
        z_ub = np.exp(-self.problem.p_lb * self.problem.b)

        z_start = np.asarray(self.lp["x"])[: self.n, 0] / SCALE
        z_start = np.maximum(z_start, z_lb)
        z_start = np.minimum(z_start, z_ub)

        x_delta, _ = self._get_x_r_delta(cube)

        x_ub = cube.center + cube.radius + 1e-12
        x_lb = cube.center - cube.radius - 1e-12

        assert all(x_delta >= x_lb)
        assert all(x_delta <= x_ub)

        theta_start = np.hstack((x_delta, z_start))

        def cnstr_1(theta, c):
            z = theta[self.m :]
            return 1 / x_lb[c] - 1 - S[c] @ z  # >= 0

        def cnstr_2(theta, c):
            z = theta[self.m :]
            return 1 + S[c] @ z - 1 / x_ub[c]  # >= 0

        constraints = [
            {"type": "ineq", "fun": lambda theta, c=c: cnstr_1(theta, c)} for c in range(self.m)
        ] + [{"type": "ineq", "fun": lambda theta, c=c: cnstr_2(theta, c)} for c in range(self.m)]

        bounds = list(zip(x_lb, x_ub)) + list(zip(z_lb, z_ub))

        CONST = 1e9

        for c in range(self.m):
            for f in [cnstr_1, cnstr_2]:
                assert (
                    f(theta_start, c) >= 0.0
                ), f"Starting value constraint violated, {f(theta_start, c)} < 0 at {f}"
        for i in range(self.n):
            assert z_start[i] <= z_ub[i], f"{i} out of bounds (> z_ub)."
            assert z_start[i] >= z_lb[i], f"{i} out of bounds ({z_start[i]} < {z_lb[i]})."

        def objective(theta):
            x, z = theta[: self.m], theta[self.m :]
            return np.sum(xlogy(z, z) * (self.problem.k @ x)) / CONST

        def jac(theta):
            x, z = theta[: self.m], theta[self.m :]
            dfdx = np.asarray([np.sum(xlogy(z, z) * self.problem.k[:, c]) for c in range(self.m)])
            z_ = np.maximum(z, z_lb)
            dfdz = (1 + np.log(z_)) * (self.problem.k @ x)
            return np.hstack((dfdx, dfdz)) / CONST

        opt = minimize(
            objective,
            theta_start,
            bounds=bounds,
            jac=jac,
            constraints=constraints,
            options={"iprint": 99, "maxiter": 10000},
        )

        assert opt.success, f"opt: {opt}, theta_start: {theta_start}."
        return -opt.fun * CONST

        # if opt.success:
        #     return -opt.fun * CONST
        # else:
        #     print("WARNING: opt for new bound failed: ", opt)
        #     return np.inf

    def compute_lipschitz_upper_bound(self, cube):

        # shift cube so that center is feasible, put in function
        x_delta, r_delta = self._get_x_r_delta(cube)

        if np.any(x_delta - r_delta < 0):
            # set upper bound high, so that cube will be branched
            cube.objective_lb = 0.0
            return np.inf

        if cube.theta_start is None:
            dual_dim = 2 * self.n + self.m
            cube.theta_start = np.random.uniform(-20.0, -50.0, size=dual_dim)

        with np.errstate(all="ignore"):
            min_problem = self._minimize_dual(cube, x_delta)

        cube.theta_start = min_problem.x
        cube.objective_lb = min_problem.fun

        dual_dx_norm_ub = self._dual_dx_norm_ub(cube, min_problem)
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

    @staticmethod
    def _dual(theta, E, u, m, n, z_lb, z_ub, kx):

        lam = theta[:m]
        exp_nu_lb = np.exp(theta[m : n + m])
        exp_nu_ub = np.exp(theta[n + m :])
        z = np.exp((E.dot(lam) + exp_nu_ub - exp_nu_lb) / kx - 1)

        return np.inner(kx, z) - np.inner(lam, u) + np.inner(exp_nu_lb, z_lb) - np.inner(exp_nu_ub, z_ub)

    @staticmethod
    def _dual_gradient(theta, E, u, m, n, z_lb, z_ub, kx):
        lam = theta[:m]
        exp_nu_lb = np.exp(theta[m : n + m])
        exp_nu_ub = np.exp(theta[n + m :])
        z = np.exp((E.dot(lam) + exp_nu_ub - exp_nu_lb) / kx - 1)
        return np.hstack([E.T.dot(z) - u, (z_lb - z) * exp_nu_lb, (z - z_ub) * exp_nu_ub,])

    def _dual_dx_norm_ub(self, cube, min_problem):
        """ Compute L(x,r) defined in the accompanying paper """
        n, m, E = self.n, self.m, self.problem.E
        x_lb, x_ub = self.problem.x_lb, self.problem.x_ub
        x, r = cube.center, cube.radius

        lam = min_problem.x[:m]
        nu_lb = np.exp(min_problem.x[m : n + m])
        nu_ub = np.exp(min_problem.x[m + n :])

        x_ = x + r
        z_arg = (lam.dot(E.T) - nu_lb + nu_ub) / np.inner(self.problem.k, x_) - 1
        z = np.exp(z_arg)
        i = lam / (
            (lam > 0) * np.square(np.maximum(x_lb, x - r)) + (lam < 0) * np.square(np.minimum(x_ub, x + r))
        ) - self.problem.w * E.T.dot(z * z_arg / self.problem.b)

        x_ = x - r
        z_arg = (lam.dot(E.T) - nu_lb + nu_ub) / np.inner(self.problem.k, x_) - 1
        z = np.exp(z_arg)
        ii = self.problem.w * E.T.dot(z * z_arg / self.problem.b) - lam / (
            (lam > 0) * np.square(np.minimum(x_lb, x + r)) + (lam < 0) * np.square(np.maximum(x_ub, x - r))
        )

        return np.max([np.max(i), np.max(ii)])

    def _minimize_dual(self, cube, x_delta, tries=500):

        u = (1 - x_delta) / x_delta
        kx = np.inner(self.problem.k, x_delta)
        theta_start = cube.theta_start
        dual_dim = 2 * self.n + self.m
        args = (self.problem.E, u, self.m, self.n, self.z_lb, self.z_ub, kx)

        for _ in range(tries):

            min_problem = minimize(
                self._dual,
                theta_start,
                args=args,
                jac=self._dual_gradient,
                method="BFGS"
            )

            if min_problem.success:
                return min_problem

            theta_start = np.random.uniform(-50.0, -20.0, size=dual_dim)

        exc = (
            f"dual optimization failed at cube.center {1 / (1 + u)}"
            f" and self.radius {self.radius}, opt: \n{min_problem}"
            f", x_lb: {self.problem.x_lb}."
        )
        raise Exception(exc)

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
        A_eq = matrix(np.hstack([E.T / SCALE, np.identity(m), -np.identity(m), np.zeros((m, 1))]))

        b_ub_1 = np.zeros(m)
        b_ub_2 = -self.z_lb * SCALE
        b_ub_3 = np.zeros(m)
        b_ub_4 = self.z_ub * SCALE
        b_ub_5 = np.zeros(m)

        b_ub = matrix(np.hstack([b_ub_1, b_ub_2, b_ub_3, b_ub_4, b_ub_5]))

        A_ub_1 = np.hstack([np.zeros((m, n)), np.identity(m), np.identity(m), -np.ones((m, 1))])
        A_ub_2 = np.hstack([-np.identity(n), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, 1))])  # z
        A_ub_3 = np.hstack([np.zeros((m, n)), -np.identity(m), np.zeros((m, m)), np.zeros((m, 1))])
        A_ub_4 = np.hstack([np.identity(n), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, 1))])  # z
        A_ub_5 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), -np.identity(m), np.zeros((m, 1))])
        A_ub_6 = np.hstack([np.zeros((m, n)), np.identity(m), np.zeros((m, m)), np.zeros((m, 1))])
        A_ub_7 = np.hstack([np.zeros((m, n)), np.zeros((m, m)), np.identity(m), np.zeros((m, 1))])
        A_ub = matrix(np.vstack([A_ub_1, A_ub_2, A_ub_3, A_ub_4, A_ub_5, A_ub_6, A_ub_7]))

        self.lp_constraints = {"c": c, "A_ub": A_ub, "A_eq": A_eq, "b_ub": b_ub}
