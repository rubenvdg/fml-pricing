from itertools import product
import logging
from multiprocessing import Pool
import time

import numpy as np
from .problem import Problem


class Cube:
    ''' Represents a hypercube during optimization '''

    def __init__(self, center, radius, x_start=None):
        self.center = center
        self.radius = radius
        self.branch = True
        self.objective_ub = None
        self.objective_lb = None
        self.objective_lower_bound_x = None
        self.x_start = x_start

    @property
    def optimality_gap(self):
        return 1 - self.objective_lb / self.objective_ub


class BranchAndBound(Problem):

    def __init__(self, *args, epsilon=0.01, max_iter=np.inf, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.logger = logging.getLogger(__name__)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.iter = 0
        self.timer = None

        self.radius = np.max(self.decision_var_ub - self.decision_var_lb) / 2
        self.cubes = [Cube(self.decision_var_lb + self.radius, self.radius)]
        self.omega = [np.asarray(arr) for arr in product([-1, 1], repeat=self.n)]
        
        self.objective_ub = np.inf
        self.objective_lb = - np.inf
        
    def compute_lower_bound(self, cube):
        raise NotImplementedError

    def compute_upper_bound(self, cube):
        raise NotImplementedError

    def solve(self):
        t0 = time.time()
        while not self.converged:
            self.iter += 1
            self.logger.debug('Iteration %s.', self.iter)
            self.logger.debug('Optimality gap %s', self.opt_gap)
            self.radius /= 2
            self.branch()
            self.logger.debug('Number of cubes %s', len(self.cubes))
            self.bound()
        self.timer = time.time() - t0
        self.logger.debug('Revenue lower bound %s', self.objective_lb)

    @property
    def opt_gap(self):
        return 1 - self.objective_lb / self.objective_ub

    @property
    def converged(self):
        
        if self.opt_gap < self.epsilon:
            self.exit_msg = f"Opt_gap = {self.opt_gap} (< epsilon)."
            return True
        
        if self.iter == self.max_iter:
            self.exit_msg = f"Maxiter reached (opt_gap = {self.opt_gap})."
            return True

        return False

    def bound(self):
        self.objective_ub = - np.inf
        for cube in self.cubes:
            self._bound_cube(cube)
    
    def _bound_cube(self, cube):
        cube.objective_lb = self.compute_lower_bound(cube)
        cube.objective_ub = self.compute_upper_bound(cube)
        if cube.objective_lb > self.objective_lb:
            self.objective_lb = cube.objective_lb
        if cube.objective_ub < self.objective_lb:
            cube.branch = False
        if cube.objective_ub > self.objective_ub:
            self.objective_ub = cube.objective_ub 

    def branch(self):
        self.cubes = [
            Cube(
                cube.center + omega * self.radius,
                self.radius,
                x_start=cube.x_start
            )
            for omega in self.omega for cube in self.cubes if cube.branch
        ]
