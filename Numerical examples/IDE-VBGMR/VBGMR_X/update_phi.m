function phi = update_phi(pi, m, v, W, beta, x, N, K, D)
phi = zeros(N, K);
for k = 1: K
    invW = inv(W(:, :, k));
    phi0 = psi(pi(k, :)) - psi(sum(pi));
    phi1 = m(k, :) * v(k, :) * invW * x';
    phi2 = sum((1 / 2) * v(k, :) * invW * x' .* x', 1);
    phi3 = (D / 2) * (1/ beta(k, :));
    phi4 = (1 / 2) * v(k, :) * m(k, :) * invW * m(k, :)';
    phi5 = (D / 2) * log(2);
    for i = 1: D
        phi60(i) = psi((v(k, :) + 1 - i) / 2);
    end
    phi6 = (1 / 2) * sum(phi60);
    phi7 = (1 / 2) * log(abs(det(W(:, :, k))));
    phi(:, k) = phi0 + phi1' - phi2' - phi3 - phi4 + phi5 + phi6 - phi7;
end
phi = softmax(phi);
% for n = 1 : N
%     phi(n, :) = softmax(phi(n, :));
% end