clc
clear all
close all

load X
Data_X0 = X(:, 1:3);
Data_Y0 = X(:, 4);

D = size(Data_X0, 2);
N0 = size(Data_Y0, 1);
DL=10;
d0=[4,5,6];
dy = zeros(d0(3), 1);
dx11 = zeros(d0(3)-d0(1), 1);
dx12 = zeros(d0(1), 1);
dx21 = zeros(d0(3)-d0(2), 1);
dx22 = zeros(d0(2), 1);
dx3 = zeros(d0(3), 1);

Data_X1 = [dx11; Data_X0(:, 1); dx12];
Data_X2 = [dx21;Data_X0(:, 2); dx22];
Data_X3 = [Data_X0(:, 3); dx3];
Data_Y1 = [dy; Data_Y0];

Data = [Data_X1,Data_X2,Data_X3,Data_Y1];

Data_X = Data(d0(3)+1:N0, 1:3);
Data_Y = Data(d0(3)+1:N0, 4);

%% 训练数据
Nb = 810;
X_tr = Data_X(1:Nb, :)';
Y_tr = Data_Y(1:Nb, :)';

%% 考虑时延，对时延进行估计，对数据进行重构
X0 = X_tr(:, 1:end);
Y = Y_tr(:, DL + 1 :end);
[Dx, Nx] = size(X0);
N = size(Y, 2);
k = Dx + size(Y, 1);
Nclst = 4;

d_DL = zeros(1, Dx);
for i = 1 : Dx
    X_S(i, :) = X0(i, DL + 1 - d_DL(i) : Nx - d_DL(i));
end
x = [X_S; Y];
X = x';

%% 1. 初始化 ---------------------------------------------------------------
% priors
alpha0 = ones(Nclst, Nclst) * 0.1;
m0 = zeros(1, k);
beta0 = 1;
W0 = eye(k) * 0.1;
v0 = 10;

beta = ones(Nclst,1);
for i=1 : Nclst
    W(:,:,i) = eye(k) * 100;
end
m = randn(Nclst, k);
v = repmat(v0, Nclst, 1);

% 随机初始化A0和A
tmp = rand(1,Nclst);
A0 = tmp/sum(tmp);
tmp = rand(Nclst, Nclst);
A = tmp./repmat(sum(tmp,2),1,Nclst);

alp_caret = zeros(Nclst, N);
c=zeros(1,N);
Pzx=zeros(Nclst,N);         % p(x|z)

bet_caret = zeros(Nclst, N);
bet_caret(:,N) = ones(Nclst, 1);

gama = zeros(Nclst,N);
xi = zeros(Nclst,Nclst,N-1);
Aold=ones(size(A))/Nclst;

%% 优化计算开始
NP=100; %% 种群大小
gen_max=30; %% 进化代数
maxIter=30;
Aerr=1e-4; % 用于迭代终止
t1 = cputime;
L=[];
for iter = 1 : maxIter
    iter
    
    [Pb,d_DL]=IDE2(Dx, DL, NP, gen_max, X0, Y, beta, m, v, W, alpha0, m0, beta0, W0, v0, A0, A, alp_caret, c, Pzx, bet_caret, gama, xi);
    
    for i = 1 : Dx
        X_S(i, :) = X0(i, DL + 1 - d_DL(i) : Nx - d_DL(i));
    end
    X = [X_S', Y'];
    
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
    lb = elbo(gama', xi, alpha_i_j, beta, v, W, alpha0, beta0, v0, W0, N, k, Nclst);
    L = [L; lb];
    
    iter_err=sum(sum(abs(A-Aold)))/Nclst;
    Aold=A;
    
end

t2 = cputime - t1;
figure
bar(d_DL);
figure
plot(L);

mu = m';
for i = 1: Nclst
    sigg(:, :, i) = W(:, :, i) / v(i, :);
end

%% 清除多余组分
r = [];
for iii = 1 : Nclst
    if sum(abs(m(iii, :)))<= 1e-6  %%% 此处条件需要先满足m0=0;
       r = [r, iii];
    end
end
mu(:, r) = [];
v(r, :) = [];
beta(r, :) = [];
sigg(:, :, r) = [];
W(:, :, r) = [];
A0(:, r) = [];
A(r, :) = [];
A(:, r) = [];
Nclst = size(mu, 2);


