# Price Optimization under the Finite-Mixture Logit model

This repository contains the implementation of the branch-and-bound algorithm described in our paper "Price Optimization under the Finite-Mixture Logit Model" (https://ssrn.com/abstract=3235432). The results in the paper are obtained on a MacBook Pro (2018) with 2.6 GHz 6-Core Intel Core i7 and 16 GB 2400 MHz DDR4 memory.

# Set up

Make a Python 3.6.8 virtual environment and run:

```
pip install -e .
pip install -r requirements.txt
```

make sure that OpenBLAS, LAPACK, and GLPK are installed on your system (on MacOS, you can use Homebrew for this).

# Reproducing the results

By running `jupyter notebook`, you can start a new notebook session and recreate the results using the following notebooks:

- Figure 4: `notebooks/figure-example-misspecification.ipynb`
- Figure 5: `notebooks/figure-cost-of-ignoring-heterogeneity.ipynb` (takes about 6 hours on the aforementioned system)
- Figure 6: `notebooks/figure-gradient-ascent-trajectories.ipynb`
- Figure 7 and 8: First run the simulation by running `python bnb/simulate.py`. The results are written to the folder `sim_results/_latest.csv`. Then run `notebooks/figure-running-times.ipynb` to produce the plots. **Note: This part is very compute and memory intensive. In `bnb/simulate.py` you can set the `REPS` parameters, which pertains to the number of simulation repetitions. In the paper we use `REPS=40`, which takes about 30 hours to run. The current value is `REPS=5`.**
- Figure 9 (from appendix): `notebooks/figure-cost-of-ignoring-segement-specific-price-sensitivity.ipynb` (takes about 30 minutes on the aforementioned system)
