function outlier_indicator = detect_outliers(res, corrupt_frac)
% input: res of size (n, K) and corrupt_frac between 0 and 1
% estimates outliers and returns an array with 1 for outlier and 0 for inlier

    n = size(res,1);
    outlier_indicator = false(1,n);
    M = min(res, [], 2); % take minimal res component per sample
    if corrupt_frac > 0 % mark outliers
        [~, outlier_supp] = maxk(M, round(corrupt_frac*n));
        outlier_indicator(outlier_supp) = true;
    end
end