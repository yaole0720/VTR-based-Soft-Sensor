function L = elbo(phi, Pi, beta, v, W, alpha0, beta0, v0, W0, N, D, K)
L0 = gammaln(sum(alpha0)) - sum(gammaln(alpha0)) - gammaln(sum(Pi)) + sum(gammaln(Pi)) - (N * D * K * log(2 * pi)) / 2;
for k = 1: K
   L10(k) = (-(v0 * D * log(2)) / 2) + ((v(k, :) * D * log(2)) / 2);
   L11(k) = -multigammaln(v0 / 2, D) + multigammaln(v(k, :) / 2, D);
   L12(k) = (D / 2) * log(abs(beta0)) - (D / 2) * log(abs(beta(k, :)));
   L13(k) = (v0 / 2) * log(det(W0)) - (v(k, :) / 2) * log(det(W(:, :, k)));
   L14(k) = L10(k) + L11(k) + L12(k) + L13(k);
end
L1 = sum(L14);
aux = zeros(N,1);
for n = 1: N
   aux(n) = log(phi(n, :)+ones(1, K)*1e-6) * phi(n, :)'; 
end
L2 = sum(aux);
L = L0 + L1 -L2;
end