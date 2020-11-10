import csv
import datetime
import json
import logging
from itertools import product
from pathlib import Path
from shutil import copyfile
from time import time

import numpy as np
from tqdm import tqdm

from bnb.fml_solver import FMLSolver
from bnb.gradient_descent import GradientDescent
from bnb.problem import OptimizationProblem

EPS = 0.01


def simulate(output_path, reps, Solver, a_range, b_range, n_range, m_range, multiprocess=True):
    logger = logging.getLogger(__name__)

    seed = 0
    t0 = time()
    _make_new_file(output_path)

    for _, m, n in tqdm(list(product(range(reps), m_range, n_range))):

        seed += 1

        logger.info("n: %s, m: %s seed: %s.", n, m, seed)

        # sample random parameters
        np.random.seed(seed)
        w = np.random.uniform(0, 1, size=m)
        w /= np.sum(w)
        a = [np.random.uniform(*a_range, size=n) for _ in range(m)]
        b = np.random.uniform(*b_range, size=n)
        problem = OptimizationProblem(a, b, w)

        # solve with gradient descent
        gd = GradientDescent(a, b, w)
        gd_sol = gd.solve()
        logger.info("gradient descent solution: %s", gd_sol)

        # solve with our solver
        solver = Solver(problem, objective_lb=gd_sol, multiprocess=multiprocess, epsilon=EPS)
        solver.solve()
        logger.info("time elapsed: %s", solver.timer)

        if gd_sol / solver.objective_lb > (1 + EPS):
            raise ValueError("Suboptimal solution.")

        # persist results
        _write_new_line(output_path, solver, seed, gd_sol)

    copyfile(output_path, output_path.parent.joinpath("_lastrun.csv"))
    logger.info("total time elapsed: %s", time() - t0)


def _make_new_file(output_path):

    with open(output_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(
            [
                "n",
                "m",
                "seed",
                "cputime",
                "iterations",
                "solver",
                "par",
                "lb",
                "ub",
                "gd_sol",
            ]
        )


def _write_new_line(output_path, solver, seed, gd_sol):
    par = {"a": list(map(list, solver.problem.A)), "b": solver.problem.b.tolist(), "w": solver.problem.w.tolist()}
    with open(output_path, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar="'")
        csvwriter.writerow(
            [
                str(solver.problem.n),
                str(solver.problem.m),
                str(seed),
                str(solver.timer),
                str(solver.iter),
                solver.__class__,
                json.dumps(par),
                solver.objective_lb,
                solver.objective_ub,
                gd_sol,
            ]
        )


if __name__ == "__main__":

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    a_range = (0.0, 7.0)
    b_range = (0.001, 0.01)
    # n_range = [10, 20, 30, 40, 50]
    # m_range = [1, 2, 3, 4]
    n_range = [30]
    m_range = [3]
    reps = 5

    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    output_path = Path("sim_results", file_name)

    simulate(output_path, reps, FMLSolver, a_range, b_range, n_range, m_range)
