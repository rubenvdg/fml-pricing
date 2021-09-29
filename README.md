# Price Optimization under the Finite-Mixture Logit model

This repository contains the implementation of the branch-and-bound algorithm described in our paper "Price Optimization under the Finite-Mixture Logit Model" (https://ssrn.com/abstract=3235432).

Set MacOS specs

I use pyenv (could also use on of the other virtual envs, conda, or a Docker container) to isolate the environment.

python 3.6.8
pip install -e .
pip install -r requirements.txt

install GLPK solver (for solving linear programs), this is what CVXOPT uses as backend (http://www.gnu.org/software/glpk/glpk.html)


Figur
e 4: `notebooks/figure-example-misspecification.ipynb`
Fast.

Figure 5: `notebooks/figure-cost-of-ignoring-heterogeneity.ipynb`
Takes quite long. About 6 hours on my laptop.

Figure 6: `notebooks/figure-gradient-ascent-trajectories.ipynb`
Fast.

Figure 7 and 8:
First run the simulation (which writes the result to ... in the folder `sim_results/`)
Then make the plots with

Figure 9 (in appendix): `notebooks/figure-cost-of-ignoring-segement-specific-price-sensitivity.ipynb`
Takes about 30 minutes.