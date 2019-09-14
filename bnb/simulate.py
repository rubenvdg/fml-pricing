import csv
import datetime
from itertools import product
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

from bnb.fml_solver import FMLSolver
from bnb.naivesolver import NaiveSolver
from bnb.problem import Problem


def simulate(path, reps):
    
    a_range = (-4.0, 4.0)
    b_range = (0.001, 0.01)
    n_range = [10, 20, 30, 40, 50]
    m_range = [2, 3]
    seed = 0
    
    csv_path = path.joinpath('results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['n', 'm', 'seed', 'cputime', 'iterations'])

    for n, m, _ in tqdm(list(product(n_range, m_range, range(reps)))):

        seed += 1
        problem = Problem(n, m, a_range, b_range, seed)
        fmlsolver = FMLSolver(problem, multiprocess=True)
        fmlsolver.solve()

        with open(csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            cpu_time = str(fmlsolver.timer)
            iters = str(fmlsolver.iter)
            csvwriter.writerow([str(n), str(m), str(seed), cpu_time, iters])

    _make_summary(csv_path)    


def _make_summary(csv_path):

    (
        pd.read_csv(csv_path)
        .melt(id_vars=['n', 'm', 'seed'])
        .groupby(['n', 'm', 'variable'])
        ['value']
        .agg(
            mean='mean',
            std='std',
            sem=st.sem,
            count='size',
        )
        .reset_index()
        .assign(
            ci=_get_ci,
            ci_lb=lambda df: df['ci'].map(lambda ci: ci[0]),
            ci_ub=lambda df: df['ci'].map(lambda ci: ci[1])
        )
        .drop(columns='ci')
        .to_csv(csv_path.parent.joinpath('summary.csv'), index=False)
    )


def _get_ci(df, q=0.95):
    return list(zip(*st.t.interval(q, df['count'] - 1, loc=df['mean'], scale=df['sem'])))


if __name__ == '__main__':
    folder_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = Path('sim_results', folder_name)
    path.mkdir()
    simulate(path=path, reps=10)

