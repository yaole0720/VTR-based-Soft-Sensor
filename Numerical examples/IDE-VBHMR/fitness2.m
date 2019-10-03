function fit_p=fitness2(d_DL, X0, Y0, DL, beta, m, v, W, alpha0, m0, beta0, v0, W0, A0, A, alp_caret, c, Pzx, bet_caret, gama, xi)

Dx = size(X0, 1);
k = Dx + 1;
Nclst = size(m, 1);
for i = 1 : Dx
    X_s(i, :) = X0(i, DL + 1 - d_DL(i) : end - d_DL(i));
end
X = [X_s', Y0'];
N = size(X, 1);

for jj = 1 : Nclst
    for ii = 1 : k
        pv(ii) = psi(v(jj)/2 + (1-ii)/2);
    end
    Pzx(jj, :) = exp((-k/2)*log(2*pi) + (k/2)*log(2) - (1/2)*log(det(W(:, :, jj))) + (1/2) * sum(pv) - (k/2)*(1/beta(jj)) - (diag((1/2)*v(jj)*(X-repmat(m(jj, :), N, 1))*inv(W(:, :, jj))*(X-repmat(m(jj, :), N, 1))'))');
end

tmp1=A0'.*Pzx(:,1);    % 实际上是alpha(z1)
c(1)=sum(tmp1);
alp_caret(:,1)=tmp1/c(1);
for kk=2:N,
    tmp1=alp_caret(:,kk-1)'*A;
    tmp2=tmp1'.*Pzx(:,kk);
    c(kk)=sum(tmp2);
    alp_caret(:,kk)=tmp2/c(kk);
end;

for kk=N:-1:2,
    tmp2=A*(Pzx(:,kk).*bet_caret(:,kk));
    bet_caret(:,kk-1)=tmp2/c(kk);
end;

gama = alp_caret.*bet_caret;
for jj=1:N-1,
    xi(:,:,jj)=alp_caret(:,jj)/c(jj+1)*(Pzx(:,jj+1).*bet_caret(:,jj+1))'.*A;
end;

Nk=sum(gama, 2);
Nk_i_j = sum(xi, 3);

alpha_i_j =alpha0 + Nk_i_j;
beta = beta0 + Nk;
v = v0 + Nk;
m = update_m(gama', beta, m0, beta0, X, k);
W = update_w(gama', beta, m, W0, beta0, m0, X, k);

A0=gama(:,1)'/sum(gama(:,1));
A = exp(psi(alpha_i_j) - repmat(psi(sum(alpha_i_j, 2)), 1, Nclst)); %% 概率转移矩阵    

% Evidence of Lower Bound
L = elbo(gama', xi, alpha_i_j, beta, v, W, alpha0, beta0, v0, W0, N, k, Nclst);
fit_p = -L;

return;