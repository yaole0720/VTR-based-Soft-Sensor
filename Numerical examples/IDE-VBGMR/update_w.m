function W = update_w(phi, beta, m, W0, beta0, m0, x, D)
K = size(phi, 2);
W = zeros(D, D, K);
for k = 1: K
   aux = (repmat(phi(:, k), 1, D) .* x)' *x;   
   W(:, :, k) = W0 + beta0 * m0' * m0 + aux - beta(k,:) * m(k, :)' * m(k, :);
end