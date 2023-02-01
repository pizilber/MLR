function [X, y_corrupted, beta, c, metrics] = generate_data(cnfg)

    % set configurations
    n = cnfg.n; d = cnfg.d; K = cnfg.K;
    overparam = cnfg.overparam;
    distrib = cnfg.distrib;
    noise_level = cnfg.noise_level;
    corrupt_frac = cnfg.corrupt_frac;

    % produce labels
    c = randsample(1:K, n, true, distrib);

    % produce matrix X
    X = randn(n,d);
    
    % produce K beta coefficients vectors
    beta = randn(d,K);

    % produce (noisy) mixed observations
    Y = zeros(n,K);
    for k=1:K
        Y(:,k) = X * beta(:,k) + noise_level * randn(n,1);
    end
    y = zeros(n,1);
    for i=1:n
        y(i) = Y(i, c(i));
    end

    % add corruptions (outliers)
    [y_corrupted, c] = corrupt(y, c, corrupt_frac);

    % evaluation metrics
    metrics.errfun = @(beta_hat) errfun_knownBeta(beta_hat, beta);
    metrics.suppfun = @(c_hat) suppfun(c_hat, c, K + max(0,overparam));
end


function [y, c] = corrupt(y, c, frac)
% add corruptions (outliers)
    n = numel(y);
    corrupt_size = round(frac*n);
    currupt_support = randsample(1:n, corrupt_size, false);
    y(currupt_support) = std(y) * randn(corrupt_size,1);
    c(currupt_support) = -1;
end
