import csv
import datetime
import json
import os
import warnings
from itertools import product
from pathlib import Path
from shutil import copyfile

import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
from time import time
import logging
from bnb.fml_solver import FMLSolver
from bnb.naivesolver import NaiveSolver
from bnb.problem import OptimizationProblem
from gradient_descent import GradientDescent

# warnings.filterwarnings("error")


def simulate(output_path, reps, Solver, a_range, b_range, n_range, m_range, multiprocess=True):

    seed = 0
    t0 = time()

    with open(output_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        columns = ["n", "m", "seed", "cputime", "iterations", "solver", "par", "lb", "ub", "gd_sol"]
        csvwriter.writerow(columns)

    for _, m, n in tqdm(list(product(range(reps), m_range, n_range))):
        print(f"n: {n}, m: {m}")

        seed += 1
        print("seed: ", seed)
        np.random.seed(seed)

        # sample random parameters
        w = np.random.uniform(0, 1, size=m)
        w /= np.sum(w)
        a = [np.random.uniform(*a_range, size=n) for _ in range(m)]
        b = np.random.uniform(*b_range, size=n)

        problem = OptimizationProblem(a, b, w)
        solver = Solver(problem, multiprocess=multiprocess, epsilon=0.01)
        print("x_lb: ", problem.x_lb)
        solver.solve()
        gd = GradientDescent(a, b, w)
        gd_sol = gd.solve()
        print("gs solution: ", gd_sol)
        print("time elapsed: ", solver.timer)

        with open(output_path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=",", quotechar="'")
            cpu_time, iters = str(solver.timer), str(solver.iter)
            par = {"a": list(map(list, a)), "b": b.tolist(), "w": w.tolist()}
            solver_name = Solver.__name__
            new_line = [
                str(n),
                str(m),
                str(seed),
                cpu_time,
                iters,
                solver_name,
                json.dumps(par),
                solver.objective_lb,
                solver.objective_ub,
                gd_sol,
            ]
            csvwriter.writerow(new_line)

    copyfile(output_path, output_path.parent.joinpath("_lastrun.csv"))
    print("time elapsed: ", time() - t0)

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    a_range = (0.0, 6.0)
    b_range = (0.001, 0.01)
    # n_range = [10, 20, 30, 40, 50]
    # m_range = [1, 2, 3, 4]
    n_range = [10]
    m_range = [4]
    reps = 1

    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    output_path = Path("sim_results", file_name)

    simulate(
        output_path=output_path,
        reps=reps,
        Solver=FMLSolver,
        a_range=a_range,
        b_range=b_range,
        n_range=n_range,
        m_range=m_range,
        multiprocess=True,
    )
