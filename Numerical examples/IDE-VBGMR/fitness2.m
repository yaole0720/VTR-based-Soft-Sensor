function fit_p=fitness2(d_DL, X0, Y0, DL, Pi, beta, m, v, W, alpha0, m0, beta0, v0, W0)

Dx = size(X0, 1);
D = Dx + 1;
K = size(Pi, 1);
for i = 1 : Dx
    X_s(i, :) = X0(i, DL + 1 - d_DL(i) : end - d_DL(i));
end
X = [X_s', Y0'];
N = size(X, 1);

phi = update_phi(Pi, m, v, W, beta, X, K, D);
Nk = sum(phi, 1)';
Pi = alpha0 + Nk;
beta = beta0 + Nk;
v = v0 + Nk;
m = update_m(phi, beta, m0, beta0, X, D);
W = update_w(phi, beta, m, W0, beta0, m0, X, D);  

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
fit_p = -L;
return;