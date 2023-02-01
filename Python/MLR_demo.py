# K-component mixture of linear regression models
# Written by Pini Zilber & Boaz Nadler / 2023

import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from auxiliary.generate_data import generate_data
from auxiliary.lib import *
from MixIRLS import MixIRLS

def main():
    # simulation configurations
    cnfg = {
        'num_experiments': 3, # number of repititions for each sample size
        'd': 100, # dimension
        'n_values': range(1000, 2000, 100), # sample size values list
        'K': 4, # number of components
        'distrib': [0.4, 0.3, 0.2, 0.1], # mixture proportions (leave empty for a perfectly balanced one)
        'noise_level': 1e-2, # Gaussian noise level
        'overparam': 0, # overparameterization
        'corrupt_frac': 0. # fraction of outliers
    }

    # MixIRLS options
    opts = {
        'rho': 1., # oversampling parameter
        'nu': 0.5, # tuning parameter
        'w_th_init': 0.1, # initialization for threshold parameter w_th
        'corrupt_frac': cnfg['corrupt_frac'], # fraction of outliers
        'unknownK': cnfg['overparam'] > 0, # if true, assumes K is unknown
        'tol': min(1, max(0.01*cnfg['noise_level'], 2e-16)) # tolerance for stopping criterion
    }

    # make sure distribution is valid
    if not cnfg['distrib']:
        cnfg['distrib'] = [(1/cnfg['K']) for k in range(cnfg['K'])]
    assert len(cnfg['distrib']) == cnfg['K'] and sum(cnfg['distrib']) - 1 < 1e-10

    # prepare result structures
    num_n_values = len(cnfg['n_values'])
    oracle_results = np.ones((num_n_values, cnfg['num_experiments']))
    MIRLS_results = oracle_results.copy()

    # run experiments
    for n_idx in range(num_n_values):
        cnfg['n'] = cnfg['n_values'][n_idx]
        print('\n********************\n \tn = ' + str(cnfg['n']) + '\n********************\n')
        
        # run realizations
        for exp_idx in range(cnfg['num_experiments']):
            print('~~~ Experiment ' + str(exp_idx) + ' ~~~')

            # genreate data
            X, y, _, c, metrics = generate_data(cnfg)
            opts['errfun'] = metrics['errfun'] # error function, used only if opts['verbose'] is true
    
            # run oracle
            oracle_beta = OLS(X, y, cnfg['K'], c)
            oracle_c = cluster_by_beta(oracle_beta, X, y, opts['corrupt_frac'])
            error = metrics['errfun'](oracle_beta)
            intersect = metrics['suppfun'](oracle_c)
            print('oracle. error: ' + '{:.3e}'.format(error) + ', \t intersection: ' + str(intersect))
            oracle_results[n_idx, exp_idx] = error

            # run Mix-IRLS
            beta_init = np.random.normal(size=(cnfg['d'], cnfg['K']+cnfg['overparam']))
            start_time = process_time()
            MIRLS_beta, iter = MixIRLS(X, y, beta_init, opts)
            time = process_time() - start_time
            MIRLS_c = cluster_by_beta(MIRLS_beta, X, y, opts['corrupt_frac'])
            error = metrics['errfun'](MIRLS_beta)
            intersect = metrics['suppfun'](MIRLS_c)
            print('MixIRLS. error: ' + '{:.3e}'.format(error) + ', \tintersection: ' + str(intersect) + ', \titer: ' + str(iter) + ', \tcpu time: ' + str(time))
            MIRLS_results[n_idx, exp_idx] = error

            print('\n***\n')
    # draw
    plt.semilogy(cnfg['n_values'], np.median(oracle_results, axis=1), linestyle='dashed', color='black')
    plt.semilogy(cnfg['n_values'], np.median(MIRLS_results, axis=1))
    plt.legend(['oracle', 'MixIRLS'])
    plt.xlabel('sample size n')
    plt.ylabel('median error')
    plt.show()

if __name__=="__main__":
   main()
        