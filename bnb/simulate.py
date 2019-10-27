import csv
import datetime
from itertools import product
import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

from bnb.fml_solver import FMLSolver
from bnb.naivesolver import NaiveSolver
from bnb.problem import OptimizationProblem


def simulate(
        output_path,
        reps,
        Solver,
        a_range,
        b_range,
        n_range,
        m_range,
        multiprocess=True):

    seed = 0    
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        columns = ['n', 'm', 'seed', 'cputime', 'iterations', 'solver']
        csvwriter.writerow(columns)

    for m, n, _ in tqdm(list(product(m_range, n_range, range(reps)))):

        seed += 1    
        np.random.seed(seed)

        # sample random parameters
        w = np.random.uniform(0, 1, size=m)
        w /= np.sum(w)
        a = [np.random.uniform(*a_range, size=n) for _ in range(m)]
        b = np.random.uniform(*b_range, size=n)
        problem = OptimizationProblem(a, b, w)
        solver = Solver(problem)
        solver.solve()

        with open(output_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            cpu_time, iters = str(solver.timer), str(solver.iter)
            solver_name = Solver.__name__
            new_line = [str(n), str(m), str(seed), cpu_time, iters, solver_name]
            csvwriter.writerow(new_line)

    copyfile(output_path, output_path.parent.joinpath('_lastrun.csv'))

if __name__ == '__main__':

    a_range = (-4.0, 4.0)
    b_range = (0.001, 0.01)
    n_range = [10, 20, 30, 40, 50]
    m_range = [1, 2, 3, 4]
    # n_range = [2, 4, 6]
    # m_range = [2, 3]
    reps = 30

    file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
    output_path = Path('sim_results', file_name)

    simulate(
        output_path=output_path,
        reps=reps,
        Solver=FMLSolver,
        a_range=a_range,
        b_range=b_range,
        n_range=n_range,
        m_range=m_range
    )

