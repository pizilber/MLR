import numpy as np
from scipy.linalg import qr
from math import ceil
from auxiliary.lib import weighted_ls, detect_outliers

def MixIRLS(X, y, beta_init, opts_input={}):
#
#   Sequential algorithm for Mixed Linear Regression
#   Written by Pini Zilber & Boaz Nadler / 2023
#
# INPUT:
# (X, y) = observations
# beta_init = initialization (K vectors)
# opts = options meta-variable
# errfun = error function w.r.t. the true beta
#
# OUTPUT:
# beta_hat = K vector estimates for beta
# iter = total iteration number (over both phases)

    ## set MixIRLS options
    # default options
    opts = dict()
    opts['verbose'] = False # let MixIRLS chat
    opts['T1'] = int(1e3) # phase I inner IRLS maximal number of iterations
    opts['T2'] = int(1e3) # phase II maximal number of iterations
    opts['wfun'] = lambda r: 1 / (1 + r**2) # IRLS function
    opts['rho'] = 1. # oversampling parameter
    opts['nu'] = 0.5 # tuning parameter
    opts['w_th_init'] = 0.1 # initialization for threshold parameter w_th
    opts['corrupt_frac'] = 0 # fraction of outliers
    opts['unknownK'] = False # if true, assumes K is unknown
    opts['tol'] = 2e-16 # tolerance for stopping criterion
    opts['errfun'] = lambda beta_hat: -1 # error function (unknown by default). Used only if verbose is on
    
    # update options according to input opts,
    # in case some option is set differently from default
    for key in opts_input.keys():
        opts[key] = opts_input[key]

    ## set parameters
    n, d = X.shape
    K = beta_init.shape[1]
    verbose = opts['verbose']
    
    ###########################
    ######### phase I #########
    ###########################
    beta_hat = np.zeros((d,K))
    supports = np.ones((n,K), dtype=np.bool8) # active samples for each component regression
    w_th = opts['w_th_init']
    first_component_w = np.zeros_like(y)

    iter = 0
    k = 0
    while k < K:
        # must also iterate for last component, as some of the active samples might
        # belong to other components, or be data outliers

        # currenet active samples
        curr_X = X[supports[:,k],:]
        curr_y = y[supports[:,k]]

        # if we repeat due to too low w_th, don't calculate first
        # component again as we can take it as is
        if k==0 and np.any(first_component_w):
            # get weights from the first component
            if verbose:
                print('use same component 1')
            w = first_component_w
            beta_hat[:, 1:] = 0
        else:
            if verbose:
                print('find component ' + str(k+1))

            # find component
            [beta, w, inner_iter] = find_component(curr_X, curr_y, opts['wfun'], \
                opts['nu'], opts['rho'], opts['T1'], beta_init[:,k], verbose)
            iter = iter + inner_iter

            beta_hat[:,k]= beta
            if k==0: # store first component in case we restart MixIRLS
                first_component_w = w
        
        next_oversampling = max(0, np.count_nonzero(w <= w_th) - opts['corrupt_frac'] * n) / d
        if not opts['unknownK'] and (k < K-1) and (next_oversampling < opts['rho']): # need more active samples
            if verbose:
                print('w_th ' + str(w_th) + ' is too low! Starting over...')
            w_th = w_th + 0.1
            k = 0
            continue
        else: # update index sets
            new_support = supports[:, k].copy()
            new_support[new_support] = (w <= w_th)
            if k < K-1: # not last component
                supports[:, k+1] = new_support

        if verbose:
            print('MixIRLS. error: ' + '{:.3e}'.format(opts['errfun'](beta_hat)) + ', \tk: ' + str(k+1))

        # If K is unknown, fix K when the next component has too few
        # active samples
        if opts['unknownK'] and (next_oversampling < opts['rho']):
            K = k+1
            beta_hat = beta_hat[:, :K]
            if verbose:
                print('MixIRLS. found K=' + str(K) + ' components, stopping here')
            break

        k = k + 1

    ###########################
    ######## phase II #########
    ###########################
    beta_diff = 1
    iter_phase2 = 0
    while (beta_diff > opts['tol']) and (iter_phase2 < opts['T2']):
        beta_hat_prev = beta_hat
        res2 = np.zeros((len(y), K))
        for k in range(K):
            res2[:, k] = abs(X @ beta_hat[:, k] - y)**2

        # caluclate weights (here a component's weight depends on the other
        # components' weight)
        w = 1 / (res2 + 1e-16)
        w = w / np.sum(w + 1e-16, axis=1)[:, np.newaxis]
        highs = np.any(w>=2/3, axis=1)
        w_highs = w[highs,:]
        w_highs[w_highs>=2/3] = 1
        w_highs[w_highs<2/3] = 0
        w[highs,:] = w_highs
        lows = np.any(w<1/K, axis=1)
        w_lows = w[lows,:]
        w_lows[w_lows<1/K] = 0
        w[lows,:] = w_lows
        w = w / np.sum(w + 1e-16, axis=1)[:, np.newaxis]

        # ignore estimated outliers
        outlier_indicator = detect_outliers(res2, opts['corrupt_frac'])
        samples_to_use = ~outlier_indicator

        # calculate new beta_hat
        for k in range(K):
                beta_hat[:, k] = weighted_ls(X[samples_to_use,:], y[samples_to_use], w[samples_to_use,k])
        beta_diff = np.linalg.norm(beta_hat - beta_hat_prev, 'fro') / np.linalg.norm(beta_hat, 'fro')
   
        # update iter and report
        iter_phase2 = iter_phase2 + 1
        if verbose and (iter_phase2 % 10 == 0):
            print('Mix-IRLS. error: ' + '{:.3e}'.format(opts['errfun'](beta_hat)) + ', \tphase2-iter: ' + str(iter_phase2))
    
    iter = iter + iter_phase2
    return beta_hat, iter


## auxiliary functions
def find_component(X, y, wfun, nu, rho, iterlim_inner, beta_init, verbose):
    # INPUT:
    # wfun = IRLS reweighting function
    # nu = tuning parameter used in IRLS reweighting function
    # rho = minimal oversampling to detect component
    # iterlim_inner = max inner iters
    # beta_init - initialization
    # OUTPUT:
    # beta = regression over large weights
    # w = final weights
    # iter = inner iters done
    
    d = X.shape[1]
    _, w, iter = MixIRLS_inner(X, y, wfun, nu, False, iterlim_inner, beta_init)
    I = np.argpartition(w, -ceil(rho * d))[-ceil(rho*d):]
    I_count = np.count_nonzero(I)
    beta = weighted_ls(X[I,:], y[I])
    if verbose:
        print('observed error: ' + str(np.linalg.norm(X[I,:] @ beta - y[I]) / np.linalg.norm(y[I])) + '. active support size: ' + str(I_count))
    return beta, w, iter


def MixIRLS_inner(X, y, wfun, nu, intercept, iterlim, beta_init=[]):
    # if beta_init is not supplied or == -1, the OLS is used
    
    n,d = X.shape
    if intercept:
        X = np.c_[np.ones(n,), X]
        d = d+1
    assert n >= d, 'not enough data'

    beta = np.zeros((d,))
    Q, R, perm = qr(X, mode='economic', pivoting=True)
    if beta_init == []:
        beta[perm,:] = weighted_ls(R, Q.T @ y)
    else:
        beta = beta_init

    # adjust residuals according to DuMouchel & O'Brien (1989)
    E = weighted_ls(R.T, X[:, perm].T).T
    h = np.sum(E * E, axis=1)
    h[h > 1 - 1e-4] = 1 - 1e-4
    adjfactor = 1 / np.sqrt(1-h)

    # IRLS
    for iter in range(iterlim):
        # residuals
        r = adjfactor * (y - X @ beta)
        rs = np.sort(np.abs(r))
        # scale
        s = np.median(rs[d:]) / 0.6745 # mad sigma
        s = max(s, 1e-6 * np.std(y)) # lower bound s in case of a good fit
        if s == 0: # perfect fit
            s = 1
        # weights
        w = wfun(r / (nu * s))
        # beta
        beta_prev = beta.copy()
        beta[perm] = weighted_ls(X[:,perm], y, w)

        # early stop if beta doesn't change
        if np.all(np.abs(beta-beta_prev) <= np.sqrt(1e-16) * np.maximum(np.abs(beta), np.abs(beta_prev))):
            break

    return beta, w, iter
