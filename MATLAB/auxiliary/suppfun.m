function intersect = suppfun(c_hat, c, K)
    % return intersection of c_hat with c

    c_hat = reshape(c_hat, size(c));
    if K < 6 % otherwise it's too computationally expensive
        perm = match_permutation_by_intersect(c_hat, c, K);
        c_hat(c_hat > 0) = perm(c_hat(c_hat > 0));  % don't rearrange outliers
        intersect = nnz(c_hat == c) / numel(c);
    else
        intersect = -1;
    end
end

function best_perm = match_permutation_by_intersect(c_hat, c, K)
% find permutation over 1:K such that c_hat best matches c

    % ignore outliers
    c(c_hat == -1) = [];
    c_hat(c_hat == -1) = [];

    % choose best permutation
    ps = perms(1:K);
    best_perm = [];
    best_intersect = -1;
    for ps_idx = 1:size(ps,1)
        curr_perm = ps(ps_idx,:)';
        curr_intersect = nnz(curr_perm(c_hat) == c);
        if curr_intersect > best_intersect
            best_intersect = curr_intersect;
            best_perm = curr_perm;
        end
    end
end