function beta = weighted_ls(X, y, w)
% weighted least squares
    d = size(X,2);
    ws = sqrt(w);
    WX = ws(:,ones(1,d)) .* X;
    wy = ws .* y;
    beta = (WX'*WX) \ (WX'*wy);
end