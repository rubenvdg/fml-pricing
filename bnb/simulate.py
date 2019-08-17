import datetime
import os
import numpy as np
import pandas as pd
from bnb import BranchAndBound


def vary_m(path, reps=50):

    for m in [2, 3, 4]:
        seed = 50
        max_iter = np.inf
        a_range = (-4.0, 4.0)
        b_range = (0.001, 0.01)
        epsilon = 0.01
        n_range = np.arange(10, 51, 10)

        for n in n_range:

            cputime = []
            iterations = []

            for _ in range(reps):

                seed += 1
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
                cputime.append(bnb.timer)
                iterations.append(bnb.iter)

            cputime_std_error = np.std(cputime) / np.sqrt(len(cputime))
            cputime = np.mean(cputime)

            iterations_std_error = np.std(iterations) / np.sqrt(len(iterations))
            iterations = np.mean(iterations)

            pd.DataFrame({
                "n": [n],
                "cputime": [cputime],
                "cputime_std_error": [cputime_std_error],
                "iterations": [iterations],
                "iterations_std_error": [iterations_std_error],
            }).to_csv(f"{path}/runtime_in_n{n}_m{m}.csv")


def vary_n(path, reps=50):
    
    for n in [10, 30, 50]:
        seed = 50
        max_iter = np.inf
        a_range = (-4.0, 4.0)
        b_range = (0.001, 0.01)
        epsilon = 0.01
        m_range = [1, 2, 3, 4]
    
        for m in m_range:

            cputime = []
            iterations = []
    
            for _ in range(reps):
    
                seed += 1
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
    
                cputime.append(bnb.timer)
                iterations.append(bnb.iter)
    
            cputime_std_error = np.std(cputime) / np.sqrt(len(cputime))
            cputime = np.mean(cputime)
    
            iterations_std_error = np.std(iterations) / np.sqrt(len(iterations))
            iterations = np.mean(iterations)
    
            pd.DataFrame({
                "m": [m],
                "cputime": [cputime],
                "cputime_std_error": [cputime_std_error],
                "iterations": [iterations],
                "iterations_std_error": [iterations_std_error],
            }).to_csv(f"{path}/runtime_in_m{m}_n{n}.csv")


if __name__ == '__main__':
    path = os.path.join('sim_results', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(path)
    vary_n(path=path, reps=2)
    vary_m(path=path, reps=2)

