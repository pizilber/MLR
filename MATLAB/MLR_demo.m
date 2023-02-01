% K-component mixture of linear regression models
% Written by Pini Zilber & Boaz Nadler / 2023
clear;
addpath(genpath('auxiliary'));

%% simulation configurations
cnfg.seed = []; % write seed number for a pseudorandom realization, or leave empty for a random one
cnfg.num_experiments = 10; % number of repititions for each sample size
cnfg.d = 100; % dimension
cnfg.n_values = [1000 : 100 : 2000]; % sample size values list
cnfg.K = 3; % number of components
cnfg.distrib = [0.7, 0.2, 0.1]; % mixture proportions (leave empty for a perfectly balanced one)
cnfg.noise_level = 1e-2; % Gaussian noise level
cnfg.overparam = 0; % overparameterization
cnfg.corrupt_frac = 0.0; % fraction of outliers

%% MixIRLS options
opts.rho = 1.; % oversampling parameter. Set to 2 in real-data experiments
opts.nu = 0.5; % tuning parameter. Set to 1 in real-data experiments
opts.w_th_init = 0.1; % initialization for threshold parameter w_th
opts.corrupt_frac = cnfg.corrupt_frac; % fraction of outliers
opts.unknownK = cnfg.overparam > 0; % if true, assumes K is unknown
opts.tol = min(1, max([0.01*cnfg.noise_level, 2*eps])); % tolerance for stopping criterion

%% prepare experiments
% make sure distribution is valid
if isempty(cnfg.distrib) % set uniform distribution
    cnfg.distrib = (1/cnfg.K) * ones(1,cnfg.K);
end
assert((numel(cnfg.distrib) == cnfg.K) && (sum(cnfg.distrib) - 1 < 1e-10), ...
    'specified distribution is invalid');

% prepare figure
fig = figure(1); clf;

% prepare result structures
num_n_values = numel(cnfg.n_values);
oracle_results = ones(num_n_values, cnfg.num_experiments);
MIRLS_results = oracle_results;

%% run experiments
for n_idx = 1:num_n_values
    cnfg.n = cnfg.n_values(n_idx);
    fprintf('\n********************\n \tn = %d \n********************\n\n', cnfg.n);
    
    if ~isempty(cnfg.seed) && (cnfg.seed > 0)
        rng(cnfg.seed);
    else
        rng('shuffle');
    end

    % run realizations
    for exp_idx = 1 : cnfg.num_experiments
        fprintf('~~~ Experiment %d ~~~\n', exp_idx);

        % genreate data
        [X, y, ~, c, metrics] = generate_data(cnfg);
        opts.errfun = metrics.errfun; % error function, used only if opts.verbose is true
 
        % run oracle
        oracle_beta = OLS(X, y, cnfg.K+cnfg.overparam, c);
        oracle_c = cluster_by_beta(oracle_beta, X, y, opts.corrupt_frac);
        error = metrics.errfun(oracle_beta);
        intersect = metrics.suppfun(oracle_c);
        fprintf('oracle. error: %d, \tintersection: %f, \n', error, intersect);
        oracle_results(n_idx, exp_idx) = error;

        % run Mix-IRLS
        beta_init = randn(cnfg.d, cnfg.K+cnfg.overparam);
        start_time = cputime;
        [MIRLS_beta, iter] = MixIRLS(X, y, beta_init, opts);
        time = cputime - start_time;
        MIRLS_c = cluster_by_beta(MIRLS_beta, X, y, opts.corrupt_frac);
        error = metrics.errfun(MIRLS_beta);
        intersect = metrics.suppfun(MIRLS_c);
        fprintf('MixIRLS. error: %d, \tintersection: %f, \titer: %d, \tcpu time: %d\n', error, intersect, iter, time);
        MIRLS_results(n_idx, exp_idx) = error;

        fprintf('\n***\n\n');
    end
end

%% draw
figure(1);
semilogy(cnfg.n_values, median(oracle_results, 2), 'k--');
hold on;
semilogy(cnfg.n_values, median(MIRLS_results, 2));
legend({'oracle', 'MixIRLS'});
xlabel('sample size n');
ylabel('median error');



%% auxiliary functions
function c_hat = cluster_by_beta(beta, X, y, corrupt_frac)
% cluster samples into components
% if robustness > 0, mark outliers with c_hat < 0
    res = abs(X*beta - y);
    outlier_indicator = detect_outliers(res, corrupt_frac);
    [~, I] = min(res, [], 2);
    I(outlier_indicator) = -1;
    c_hat = I;
end

function beta_hat = OLS(X, y, K, c)
    % if c is provided, perform OLS to each component individually according to c
    % otherwise, perform a standard (single-component) OLS
    if nargin < 4
        beta_hat = repmat(X\y, 1, K);
    else
        beta_hat = zeros(size(X,2),K);
        for k=1:K
            beta_hat(:,k) = X(c==k,:) \ y(c==k);
        end
    end
end
