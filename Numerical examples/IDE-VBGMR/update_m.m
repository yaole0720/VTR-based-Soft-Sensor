function m = update_m(phi, beta, m0, beta0, x, D)
K = size(phi, 2);
m = zeros(K, D);
for k = 1: K
    m(k, :) = (m0 * beta0 + phi(:, k)' * x) / (beta(k, :));
end
