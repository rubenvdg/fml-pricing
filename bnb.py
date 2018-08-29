import numpy as np
from numpy.linalg import norm
import time
from scipy.optimize import minimize, root
from cvxopt import matrix, solvers
from multiprocessing import Pool

solvers.options['show_progress'] = False


class Segment:
    ''' Represents a customer segment  '''

    def __init__(self, a_range, b, n, share):
        self.a = np.random.uniform(a_range[0], a_range[1], size=n)
        self.b = b
        self.share = share

    def compute_bounds_x(self, p_lb, p_ub):
        self.x_lb = 1 / (1 + np.sum(np.exp(self.a - self.b * p_lb)))
        self.x_ub = 1 / (1 + np.sum(np.exp(self.a - self.b * p_ub)))


class Cube:
    ''' Represents a hypercube in during optimization '''

    def __init__(self, center, radius, theta_start=None):

        self.feasible = False
        self.remove = False
        self.center = center
        self.radius = radius
        self.theta_start = theta_start
        self.x_delta = None
        self.r_delta = None
        self.D = None
        self.dDdx_norm_ub = None
        self.rev_ub = None


class BranchAndBound:
    '''
    Represents the branch-and-bound algorithm from the paper
    "Price Optimization Under the Finite-Mixture Logit Model" (2018)
    by R. van de Geer and A. V. den Boer
    available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3235432
    '''

    def __init__(self, n, m, seed, max_iter, a_range, b_range, epsilon):

        self.iter = 0
        self.n = n
        self.m = m
        self.seed = seed
        self.max_iter = max_iter
        self.a_range = a_range
        self.b_range = b_range
        self.epsilon = epsilon

        if self.seed:
            np.random.seed(self.seed)

        self.w = np.random.uniform(0, 1, size=m)
        self.w /= np.sum(self.w)
        self.b = np.random.uniform(b_range[0], b_range[1], size=n)
        self.segments = [Segment(a_range, self.b, n, wc) for wc in self.w]

        self.radius = None
        self.timer = None

        # self.omega = {-1, 1}^m
        self.omega = np.vstack(
            map(np.ravel, np.meshgrid(*([[-1, 1] for _ in range(self.m)])))
        ).T

        self.compute_bounds_p_x()
        self.z_lb = np.exp(-self.p_ub * self.b)
        self.z_ub = np.exp(-self.p_lb * self.b)
        self.x_ub = np.asarray([segment.x_ub for segment in self.segments])
        self.x_lb = np.asarray([segment.x_lb for segment in self.segments])
        self.radius = np.max(self.x_ub - self.x_lb) / 2
        self.E = np.asarray([np.exp(segment.a) for segment in self.segments]).T
        self.k = self.E * self.w.reshape(1, -1) / self.b.reshape(-1, 1)

    @staticmethod
    def pi_c(pi, segment, b):
        return pi - np.sum(1 / b * np.exp(segment.a - 1 - b * pi))

    @staticmethod
    def D(theta, E, u, m, n, z_lb, z_ub, kx):

        lam = theta[:m]
        exp_nu_lb = np.exp(theta[m:n+m])
        exp_nu_ub = np.exp(theta[n+m:])
        z = np.exp((E.dot(lam) + exp_nu_ub - exp_nu_lb) / kx - 1)

        return (np.inner(kx, z)
                - np.inner(lam, u)
                + np.inner(exp_nu_lb, z_lb)
                - np.inner(exp_nu_ub, z_ub))

    @staticmethod
    def Dgrad(theta, E, u, m, n, z_lb, z_ub, kx):
        lam = theta[:m]
        exp_nu_lb = np.exp(theta[m:n+m])
        exp_nu_ub = np.exp(theta[n+m:])
        z = np.exp((E.dot(lam) + exp_nu_ub - exp_nu_lb) / kx - 1)
        return np.hstack([
            E.T.dot(z) - u,
            (z_lb - z) * exp_nu_lb,
            (z - z_ub) * exp_nu_ub,
        ])

    def compute_bounds_p_x(self):

        # compute upper bound on segment revenues
        ub_pi = []
        for segment in self.segments:
            res = root(self.pi_c, 5.0, args=(segment, self.b))
            if not res.success:
                raise Exception('Root finding failed.')
            else:
                ub_pi.append(res.x)

        # compute upper bound on prices
        self.p_ub = 1 / self.b + np.max(ub_pi)
        self.p_lb = 1 / self.b

        # compute bounds on prices
        for segment in self.segments:
            segment.compute_bounds_x(self.p_lb, self.p_ub)

    def check_feasibility_cube_cvx(self, cube, scale_r=1.0):

        m = self.m
        n = self.n
        x = cube.center
        E = self.E
        u = (1 - x) / x
        r = self.radius

        c = matrix(np.hstack([np.zeros(n + 2 * m), np.ones(1)]))

        b_eq = matrix(u)
        A_eq = matrix(np.hstack([E.T,
                                 np.identity(m),
                                 -np.identity(m),
                                 np.zeros((m, 1))]))

        b_ub_1 = np.zeros(m)
        A_ub_1 = np.hstack([np.zeros((m, n)),
                            np.identity(m),
                            np.identity(m),
                            -np.ones((m, 1))])

        b_ub_2 = u - (1 - (x + r)) / (x + r)
        A_ub_2 = np.hstack([np.zeros((m, n)),
                            np.identity(m),
                            np.zeros((m, m)),
                            np.zeros((m, 1))])

        b_ub_3 = np.zeros(m)
        A_ub_3 = np.hstack([np.zeros((m, n)),
                            - np.identity(m),
                            np.zeros((m, m)),
                            np.zeros((m, 1))])

        b_ub_4 = (1 - (x - r)) / (x - r) - u
        A_ub_4 = np.hstack([np.zeros((m, n)),
                            np.zeros((m, m)),
                            np.identity(m),
                            np.zeros((m, 1))])

        b_ub_5 = np.zeros(m)
        A_ub_5 = np.hstack([np.zeros((m, n)),
                            np.zeros((m, m)),
                            - np.identity(m),
                            np.zeros((m, 1))])

        b_ub_6 = - self.z_lb
        A_ub_6 = np.hstack([-np.identity(n),
                            np.zeros((n, m)),
                            np.zeros((n, m)),
                            np.zeros((n, 1))])

        b_ub_7 = self.z_ub
        A_ub_7 = np.hstack([np.identity(n),
                            np.zeros((n, m)),
                            np.zeros((n, m)),
                            np.zeros((n, 1))])

        A_ub = matrix(np.vstack([
            A_ub_1, A_ub_2, A_ub_3, A_ub_4, A_ub_5, A_ub_6, A_ub_7]))
        b_ub = matrix(np.hstack([
            b_ub_1, b_ub_2, b_ub_3, b_ub_4, b_ub_5, b_ub_6, b_ub_7]))

        return solvers.lp(c, A_ub, b_ub, A_eq, b_eq,
                          solver='glpk',
                          options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})

    def dDdx_norm_ub(self, cube, nu_lb, nu_ub, lam):
        ''' Compute L(x,r) defined in the accompanying paper '''

        E = self.E
        x = cube.center
        r = cube.radius

        x_ = x + r
        z_arg = (lam.dot(E.T) - nu_lb + nu_ub) / np.inner(self.k, x_) - 1
        z = np.exp(z_arg)
        i = (lam / ((lam > 0) * np.square(np.maximum(self.x_lb, x - r)) +
                    (lam < 0) * np.square(np.minimum(self.x_ub, x + r)))
             - self.w * E.T.dot(z * z_arg / self.b))

        x_ = x - r
        z_arg = (lam.dot(E.T) - nu_lb + nu_ub) / np.inner(self.k, x_) - 1
        z = np.exp(z_arg)
        ii = (self.w * E.T.dot(z * z_arg / self.b)
              - lam / ((lam > 0) * np.square(np.minimum(self.x_lb, x + r)) +
                       (lam < 0) * np.square(np.maximum(self.x_ub, x - r))))

        return np.max([np.max(i), np.max(ii)])

    def minimize_D(self, cube):

        success = False
        u = (1 - cube.x_delta) / cube.x_delta
        kx = np.inner(self.k, cube.x_delta)
        count_tries = 0
        theta_start = cube.theta_start
        method = 'BFGS'

        while not success:

            count_tries += 1
            args = (self.E, u, self.m, self.n, self.z_lb, self.z_ub, kx)
            out = minimize(self.D, theta_start,
                           args=args, jac=self.Dgrad, method=method)

            if out.success:
                success = True
            elif count_tries > 10 and out.status != 2:
                exc = (f"lagrange optimization failed at " +
                       f"cube.center {1/(1+u)}" +
                       f" and self.radius {self.radius}, opt: \n{out}")
                raise Exception(exc)

            if count_tries == 1:
                method = 'BFGS'
            else:
                theta_start = np.random.uniform(-20.0, -50.0,
                                                size=2*self.n+self.m)

        return out

    def iterate_cubes(self, cubes):

        n = self.n
        m = self.m

        if self.iter > 1:

            cubes = [
                Cube(cube.center + omega * self.radius,
                     self.radius,
                     theta_start=cube.theta_start)
                for omega in self.omega for cube in cubes
            ]

        for cube in cubes:

            # remove cubes that are completely outside [x_lb, x_ub]

            if np.any(cube.center - cube.radius > self.x_ub):
                cube.remove = True

            # check if feasible for the other cubes

            else:

                feasibility_cube_cvx = self.check_feasibility_cube_cvx(cube)
                cube.feasible = (True if feasibility_cube_cvx['status'] ==
                                 'optimal' else False)

                if cube.feasible:

                    delta_plus = np.asarray(
                        feasibility_cube_cvx['x'])[n:n+m, 0]
                    delta_min = np.asarray(
                        feasibility_cube_cvx['x'])[n+m:n+2*m, 0]

                    u = ((1 - cube.center) / cube.center -
                         delta_plus + delta_min)
                    cube.x_delta = 1 / (1 + u)
                    cube.r_delta = cube.radius + norm(cube.center -
                                                      cube.x_delta, ord=np.inf)

                    if np.any(cube.x_delta - cube.r_delta < 0):
                        cube.rev_ub = np.inf
                        cube.D = 0.0
                    else:
                        with np.errstate(all='ignore'):
                            opt = self.minimize_D(cube)
                        cube.theta_start = opt.x
                        lam = opt.x[:m]
                        nu_lb = np.exp(opt.x[m:n+m])
                        nu_ub = np.exp(opt.x[m+n:])
                        cube.D = opt.fun
                        cube.dDdx_norm_ub = self.dDdx_norm_ub(cube, nu_lb,
                                                              nu_ub, lam)
                        cube.rev_ub = cube.D + cube.dDdx_norm_ub * cube.r_delta

                    if cube.rev_ub < self.rev_lb:
                        cube.suboptimal = True
                        cube.remove = True

                else:
                    cube.remove = True

        return cubes

    def bnb(self):

        t0 = time.time()
        self.rev_lb = 0.0

        # initialize cubes
        theta_start = np.random.uniform(-20.0, -50.0, size=2*self.n+self.m)
        self.cubes = [Cube(self.x_lb + self.radius, self.radius,
                           theta_start=theta_start)]

        stop = False

        while not stop:

            self.iter += 1

            if self.m > 2:
                with Pool() as pool:
                    self.cubes = [cube for cubes in pool.map(
                                    self.iterate_cubes,
                                    np.array_split(self.cubes, pool._processes)
                                  ) for cube in cubes]
            else:
                self.cubes = self.iterate_cubes(self.cubes)

            self.cubes = [cube for cube in self.cubes if not cube.remove]

            self.rev_lb = np.max(
                [np.max([cube.D for cube in self.cubes]), self.rev_lb]
            )
            rev_ub = np.max([cube.rev_ub for cube in self.cubes])
            opt_gap = 1 - self.rev_lb / rev_ub

            if opt_gap < self.epsilon:
                self.exit_msg = "Exit because optimality gap < epsilon."
                stop = True
            elif self.iter == self.max_iter:
                self.exit_msg = (f"Exit because maxiter reached, " +
                                 f"current optimality gap: {opt_gap:.4f}.")
                stop = True
            else:
                self.radius *= 0.5

            print(f"iteration: {self.iter}, " +
                  f"cube count before branching: {len(self.cubes)}, " +
                  f"opt_gap: {opt_gap:.4f}, rev_lb: {self.rev_lb:.4f}")

        self.timer = time.time() - t0


if __name__ == '__main__':

    m = 3
    n = 25
    max_iter = np.inf
    a_range = (-4.0, 4.0)
    b_range = (0.001, 0.01)
    epsilon = 0.01
    seed = 1

    bnb = BranchAndBound(
        n=n,
        m=m,
        seed=seed,
        max_iter=max_iter,
        a_range=a_range,
        b_range=b_range,
        epsilon=epsilon,
    )

    bnb.bnb()
    print(bnb.exit_msg)
    print(f"Time elapsed: {bnb.timer:.2f}")
