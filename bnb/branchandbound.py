from itertools import product
import logging
from multiprocessing import Pool
import time
import numpy as np


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


class BranchAndBound:

    def __init__(self, bounds, epsilon=0.01, multiprocess=True):
        
        self.logger = logging.getLogger(__name__)
        self.epsilon = epsilon
        self.iter = 0
        self.multiprocess = multiprocess
        
        lower_bound, upper_bound = bounds
        dim = len(lower_bound)
        self.radius = np.max(upper_bound - lower_bound) / 2
        self.cubes = [Cube(lower_bound + self.radius, self.radius)]
        self.omega = [np.asarray(arr) for arr in product([-1, 1], repeat=dim)]

        self.objective_ub = np.inf
        self.objective_lb = - np.inf
        self.timer = None

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
        return False

    def bound(self):
        
        if self.multiprocess:
            with Pool() as pool:
                self.cubes = [
                    cube
                    for cubes in pool.map(
                        self._bound_cubes,
                        np.array_split(self.cubes, pool._processes)
                    )
                    for cube in cubes
                ]
        else: 
            self.cubes = self._bound_cubes(self.cubes)

        self.objective_ub = - np.inf
        for cube in self.cubes:
            if cube.objective_lb > self.objective_lb:
                self.objective_lb = cube.objective_lb
            if cube.objective_ub < self.objective_lb:
                cube.branch = False
            if cube.objective_ub > self.objective_ub:
                self.objective_ub = cube.objective_ub 
    
    def _bound_cube(self, cube):
        cube.objective_lb = self.compute_lower_bound(cube)
        cube.objective_ub = self.compute_upper_bound(cube)
        return cube
    
    def _bound_cubes(self, cubes):
        return [self._bound_cube(cube) for cube in cubes]

    def branch(self):
        self.cubes = [
            Cube(
                cube.center + omega * self.radius,
                self.radius,
                x_start=cube.x_start
            )
            for omega in self.omega for cube in self.cubes if cube.branch
        ]
