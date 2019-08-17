from itertools import product
import numpy as np


class Cube:
    ''' Represents a hypercube in during optimization '''

    def __init__(self, center, radius):

        self.remove = False
        self.center = center
        self.radius = radius
        self.objective_upper_bound = None


# solver inherits branch and bound and sets local bounding functions

class BranchAndBound:

    def __init__(self, x_lb: np.ndarray, x_ub: np.ndarray):

        assert x_lb.shape == x_ub.shape
        assert len(x_lb.shape) == 1
        
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.n = len(x_lb)
        self.radius = np.max(x_ub - x_lb) / 2
        self.cubes = [Cube(x_lb + self.radius, self.radius)]
        self.omega = [np.asarray(arr) for arr in product([-1, 1], repeat=self.n)]
        self.objective_upper_bound = np.inf

    @staticmethod
    def compute_lower_bound(cube):
        raise NotImplementedError

    def bound(self):

        for cube in self.cubes:
            cube.lower_bound = self.compute_lower_bound(cube)


    def branch(self):
        pass

    
    












class BranchBound:
    def __init__(self, x_lower_bound, x_upper_bound):
        self.x_lower_bound = x_lower_bound
        self.x_upper_bound = x_upper_bound
        ...
    
    def compute_upper_bound(self):
        pass

    def bound(self):
        self.compute_upper_bound()

class GoodSolver(BranchBound):
    def compute_upper_bound(self):
        do_good_stuff_here():


        

