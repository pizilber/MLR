# Mixed Linear Regression (MLR)
MLR is a simple generalization of linear regression. In MLR, each sample belongs to one of $K$ unknown linear models, and it is not known to which one. MLR can thus be viewed as a combination of linear regression and clustering. A book reference can be found in Bishop (2006), Pattern Recognition, Chapter 14.

# Mixed-IRLS
This repository contains `MATLAB` and `Python` implementations for the `Mixed-IRLS` algorithm, described in [this arXiv preprint](https://arxiv.org/abs/2301.12559).
The key idea that distinguishes `Mixed-IRLS` from most other methods is sequential recovery: rather than recoverying all the $K$ models simultaneously, `Mixed-IRLS` recovers them one by one, using tools from robust regression. Specifically, it uses iteratively reweighted least squares (IRLS) with a random initialization.
Importantly, in its internediate steps, `Mix-IRLS` allows an "I don't know" assignment to low-confidence samples, namely samples whose model identity is uncertain. These samples are ignored, and used only in a later refinement phase. See an illustration of `Mix-IRLS` in the figure below.

`Mixed-IRLS` deals well with balanced mixtures, where the proportions of the $K$ components are approximately equal; however, it is particulary effective for imbalanced mixtures. In addition, `Mixed-IRLS` can handle outliers in the data, and overestimated/unknown number of components $K$.

![MixIRLS illustration](https://github.com/pizilber/MLR/blob/main/MixIRLS_illustration.png)

## Usage
Simple demos demonstrating the usage of the algorithms are available, see `MLR_demo.m` and `MLR_demo.py`.
The simulation depends on the following configuration parameters:
- d: problem dimension
- n_values: list of sample sizes
- K: true number of components
- distrib: mixture proportions
- noise_level: Gaussian noise standard deviation
- overparam: overparameterization, such that the algorithm gets K+overparam as input
- corrupt_frac: fraction of outliers

`Mix-IRLS` depends on the following parameters, described in [the manuscript](https://arxiv.org/abs/2301.12559):
- rho $\geq 1$: oversampling ratio. Default value: $1$ for synthetic data, $2$ for real data
- $0 <$ w_th_init $< 1$: weight threshold. Default value: $0.01$
- nu $> 0$: tuning parameter. Default value: $0.5$ for synthetic data, $1$ for real data
- $0 <$ corrupt_frac $< 1$: outlier fraction estimate. Default value: $0$
- unknownK: true if $K$ is unknown, false otherwise. Default value: false
- tol: stopping criterion tolerance. Default value: $\min(1, \max(1, 0.01 \cdot \text{noise-level}, 2\epsilon_\text{machine-precision}))$

## Citation
If you refer to the `Mix-IRLS` method or the manuscript, please cite them as:
```
@article{zilber2023imbalanced,
  title={Imbalanced Mixed Linear Regression},
  author={Zilber, Pini and Nadler, Boaz},
  journal={arXiv preprint arXiv:2301.12559},
  year={2023}
}
```
