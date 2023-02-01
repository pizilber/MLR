function [beta_hat, iter] = MixIRLS(X, y, beta_init, opts_input)
%
%   Sequential algorithm for Mixed Linear Regression
%   Written by Pini Zilber & Boaz Nadler / 2023
%
% INPUT:
% (X, y) = observations
% beta_init = initialization (K vectors)
% opts = options meta-variable
% errfun = error function w.r.t. the true beta
%
% OUTPUT:
% beta_hat = K vector estimates for beta
% iter = total iteration number (over both phases)

    %% set MixIRLS options
    % default options
    opts.verbose = false; % let MixIRLS chat
    opts.T1 = 1e3; % phase I inner IRLS maximal number of iterations
    opts.T2 = 1e3; % phase II maximal number of iterations
    opts.wfun = @(r) 1 ./ (1 + r.^2); % IRLS function
    opts.rho = 1.; % oversampling parameter
    opts.nu = 0.5; % tuning parameter
    opts.w_th_init = 0.1; % initialization for threshold parameter w_th
    opts.corrupt_frac = 0; % fraction of outliers
    opts.unknownK = false; % if true, assumes K is unknown
    opts.tol = 2*eps; % tolerance for stopping criterion
    opts.errfun = @(beta_hat) -1; % error function (unknown by default). Used only if verbose is on
    
    % update options according to input opts,
    % in case some option is set differently from default
    if nargin > 3
        for fn = fieldnames(opts_input)'
           opts.(fn{1}) = opts_input.(fn{1});
        end
    end

    %% set parameters
    [n,d] = size(X);
    K = size(beta_init,2);
    verbose = opts.verbose;
    
    %% phase I
    beta_hat = zeros(d,K);
    supports = true(n,K); % active samples for each component regression
    w_th = opts.w_th_init;
    first_component_w = zeros(size(y));

    iter = 0;
    k = 1;
    while k <= K
        % must also iterate for k==K, as some of the active samples might
        % belong to other components, or be data outliers

        % currenet active samples
        curr_X = X(supports(:,k),:);
        curr_y = y(supports(:,k));

        % if we repeat due to too low w_th, don't calculate first
        % component again as we can take it as is
        if (k==1) && any(first_component_w)
            % get weights from the first component
            if verbose
                fprintf('use same component 1\n');
            end
            w = first_component_w;
            beta_hat(:, 2:end) = 0;
        else
            if verbose
                fprintf('find component %d\n', k);
            end

            % find component
            [beta, w, inner_iter] = find_component(...
                curr_X, curr_y, opts.wfun, opts.nu, opts.rho, opts.T1, beta_init(:,k), verbose);
            iter = iter + inner_iter;

            beta_hat(:,k) = beta;
            if k==1 % store first component in case we restart MixIRLS
                first_component_w = w;
            end
        end
        
        next_oversampling = max(0, sum(w<=w_th) - opts.corrupt_frac*n) / d;
        if ~opts.unknownK && (k < K) && (next_oversampling < opts.rho) % need more active samples
            if verbose
                fprintf('w_th %f is too low! Starting over...\n', w_th);
            end
            w_th = w_th + 0.1;
            k = 1;
            continue;
        else % update index sets
            new_support = supports(:, k);
            new_support(new_support) = (w <= w_th);
            if k < K
                supports(:, k+1) = new_support;
            end
        end

        if verbose
            fprintf('MixIRLS. error: %d, \tk: %d\n', opts.errfun(beta_hat), k);
        end

        % If K is unknown, fix K when the next component has too few
        % active samples
        if opts.unknownK && (next_oversampling < opts.rho)
            K = k;
            beta_hat = beta_hat(:, 1:K);
            if verbose
                fprintf('MixIRLS. found K=%d components, stopping here\n', k);
            end
            break;
        end

        k = k + 1;
    end

    %% phase II
    beta_diff = 1;
    iter_phase2 = 0;
    while (beta_diff > opts.tol) && (iter_phase2 < opts.T2)
        beta_hat_prev = beta_hat;
        res2 = abs(X*beta_hat - y).^2;

        % caluclate weights (here a component's weight depends on the other
        % components' weight)
        w = 1 ./ (res2 + eps);
        w = w ./ repmat(sum(w + eps, 2), [1, K]);
        highs = any(w>=2/3, 2);
        w_highs = w(highs,:);
        w_highs(w_highs>=2/3) = 1;
        w_highs(w_highs<2/3) = 0;
        w(highs,:) = w_highs;
        lows = any(w<1/K, 2);
        w_lows = w(lows,:);
        w_lows(w_lows<1/K) = 0;
        w(lows,:) = w_lows;
        w = w ./ repmat(sum(w + eps, 2), [1, K]);

        % ignore estimated outliers
        outlier_indicator = detect_outliers(res2, opts.corrupt_frac);
        samples_to_use = ~outlier_indicator;

        % calculate new beta_hat
        for k = 1:K
                beta_hat(:, k) = weighted_ls(X(samples_to_use,:), y(samples_to_use), w(samples_to_use,k));
        end
        beta_diff = norm(beta_hat - beta_hat_prev, 'fro') / norm(beta_hat, 'fro');
   
        % update iter and report
        iter_phase2 = iter_phase2 + 1;
        if verbose && (mod(iter_phase2, 10) == 0)
            fprintf('Mix-IRLS. error: %d, \tphase2-iter: %d\n', opts.errfun(beta_hat), iter_phase2);
        end
    end

    iter = iter + iter_phase2;
end


%% auxiliary functions
function [beta, w, iter] = find_component(X, y, wfun, nu, ...
    rho, iterlim_inner, beta_init, verbose)
    % INPUT:
    % wfun = IRLS reweighting function
    % nu = tuning parameter used in IRLS reweighting function
    % rho = minimal oversampling to detect component
    % iterlim_inner = max inner iters
    % beta_init - initialization
    % OUTPUT:
    % beta = regression over large weights
    % w = final weights
    % iter = inner iters done
    
    d = size(X,2);
    [~, w, iter] = MixIRLS_inner(X, y, wfun, nu, false, iterlim_inner, beta_init);
    [~, I] = maxk(w, ceil(rho * d));
    beta = X(I,:) \ y(I);
    if verbose
        fprintf('observed error: %f. active support size: %d\n', ...
            norm(X(I,:) * beta - y(I)) / norm(y(I)), nnz(I));
    end
end


function [beta, w, iter] = MixIRLS_inner(X, y, wfun, nu, intercept, iterlim, beta_init)
    % based on MATLAB's statrobustfit implementation
    % if beta_init is not supplied or == -1, the OLS is used
    
    [n,d] = size(X);
    if intercept
        X = [ones(n,1), X];
        d = d+1;
    end
    assert(n >= d, 'not enough data');

    [Q, R, perm] = qr(X,0);
    if (nargin < 7) || isempty(beta_init)
        beta(perm,:) = R \ (Q'*y);
    else
        beta = beta_init;
    end

    % adjust residuals according to DuMouchel & O'Brien (1989)
    E = X(:,perm) / R;
    h = min(1 - 1e-4, sum(E.*E, 2));
    adjfactor = 1 ./ sqrt(1-h);

    % IRLS
    for iter=1:iterlim
        % residuals
        r = adjfactor .* (y - X*beta);
        rs = sort(abs(r));
        % scale
        s = median(rs(d:end)) / 0.6745; % mad sigma
        s = max(s, 1e-6 * std(y)); % lower bound s in case of a good fit
        if s == 0 % perfect fit
            s = 1;
        end
        % weights
        w = feval(wfun, r / (nu * s));
        % beta
        beta_prev = beta;
        beta(perm) = weighted_ls(X(:,perm), y, w);

        % early stop if beta doesn't change
        if all(abs(beta-beta_prev) <= sqrt(eps) * max(abs(beta),abs(beta_prev)))
            break;
        end
    end
end
