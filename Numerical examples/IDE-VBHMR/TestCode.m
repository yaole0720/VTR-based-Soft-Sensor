clc
clear all
close all

load Work_TR

%% 测试数据
Nb2 = 1020;
X_Ts = Data_X(Nb+1 : Nb2, :)';
Y_Ts = Data_Y(Nb+1 : Nb2, :)';

% X_Ts = X_tr;
% Y_Ts = Y_tr;
Nts = size(X_Ts, 2);
for i = 1 : Dx
    X_TS(i, :) = X_Ts(i, DL + 1 - d_DL(i) : Nts - d_DL(i));
end
Y_TS = Y_Ts(:, DL + 1 : end);
xt = [X_TS; Y_TS]';
Nt = size(xt, 1);

m = mu';
for jj = 1 : Nclst
    for ii = 1 : k
        pv(ii) = psi(v(jj)/2 + (1-ii)/2);
    end
    Pzxt(jj, :) = exp((-k/2)*log(2*pi) + (k/2)*log(2) - (1/2)*log(det(W(:, :, jj))) + (1/2) * sum(pv) - (k/2)*(1/beta(jj)) - (diag((1/2)*v(jj)*(xt - repmat(m(jj, :), Nt, 1))*inv(W(:, :, jj))*(xt - repmat(m(jj, :), Nt, 1))'))');
end
bet_carett = zeros(Nclst, Nt);
bet_carett(:,Nt) = ones(Nclst, 1);
tmpt1=A0'.*Pzxt(:,1);    % 实际上是alpha(z1)
ct(1)=sum(tmpt1);
alp_carett(:,1)=tmpt1/ct(1);
for kk=2:Nt,
    tmpt1=alp_carett(:,kk-1)'*A;
    tmpt2=tmpt1'.*Pzxt(:,k);
    ct(kk)=sum(tmpt2);
    alp_carett(:,kk)=tmpt2/ct(kk);
end;

for kk=Nt:-1:2,
    tmpt2=A*(Pzxt(:,kk).*bet_carett(:,kk));
    bet_carett(:,kk-1)=tmpt2/ct(kk);
end;

gamat = alp_carett.*bet_carett;

%%% 测试
for i = 1: Nclst
    S_in(:, :, i) = sigg(1: k - 1, 1: k - 1, i);
    S_out(:, :, i) = sigg(k, k, i);
    S_oi(:, :, i) = sigg(k, 1: k - 1, i);
end
for i = 1: Nclst
    miu_in(i, :) = m(i, 1: k - 1);
    miu_out(i, :) = m(i, k);
end

xt_in = xt(:, 1: k-1);
xt_out = xt(:, k);
for t = 1: Nt
    for i = 1: Nclst
        miu_t_out(t, i) = miu_out(i, :) + S_oi(:, :, i) * inv(S_in(:, :, i)) * (xt_in(t, :) - miu_in(i, :))';
    end
    Hx(:, t) = Pzxt(:, t) / sum(Pzxt(:, t), 1);
    xt_out_pre(t,:) = Hx(:, t)' * miu_t_out(t, :)';
end

RMSE = sqrt((xt_out - xt_out_pre)' * (xt_out - xt_out_pre) / Nt)
R2 = 1 - (sum((xt_out - xt_out_pre) .* (xt_out - xt_out_pre))) / (sum((xt_out - mean(xt_out)) .* (xt_out - mean(xt_out))))

figure
plot(xt_out, 'b.-');
hold on
plot(xt_out_pre, 'r.-');