% calculates the error of an estimate beta_hat, invariant to the order of components
% can handle also overparameterized beta_hat
function err = errfun_knownBeta(beta_hat, beta)
    K = size(beta, 2);

    if size(beta_hat,2) < K % in case of underparameterization, zero-pad
        beta_hat = [beta_hat, zeros(size(beta,1), K-size(beta_hat,2))];
    end

    ps = perms(1:K);
    err = Inf;
    for ps_idx = 1:size(ps,1)
        % beta_temp of the first K components of current permutation
        beta_temp = beta_hat(:, ps(ps_idx, :));
        beta_temp = beta_temp(:, 1:K);
        K_errors = vecnorm(beta_temp - beta);
        curr_err = mean(K_errors);
        if curr_err < err
            err = curr_err;
        end
    end
end

