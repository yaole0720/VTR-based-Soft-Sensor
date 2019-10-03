clc
clear all
close all

load X1
Data_X0 = X1(:, 1:3);
Data_Y0 = X1(:, 4);

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

X0 = X_tr(:, 1:end);
Y = Y_tr(:, DL + 1 :end);
[Dx, Nx] = size(X0);
N = size(Y, 2);
D = Dx + size(Y, 1);
K = 3;

%% 初始化
% priors
alpha0 = ones(K,1);
m0 = zeros(1,D);
beta0 = 1;
W0 = eye(D)*0.1;
v0 = 10;

% initialize variational hyper-parameters
Pi = [0.3; 0.3; 0.4];
beta = ones(K,1);
for i=1:K
    W(:,:,i)=eye(D) * 100;
end
m = randn(K, D);
v = repmat(v0,K,1);

%% IDE 迭代  有VTR模型
NP=100; %% 种群大小
gen_max=30; %% 进化代数
maxIter=30;
t1 = cputime;
L=[];
n_iters=0;
for iter=1:maxIter
    iter    
   
    [Pb,d_DL]=IDE2(Dx, DL, NP, gen_max, X0, Y, Pi, beta, m, v, W, alpha0, m0, beta0, W0, v0);
    for i = 1 : Dx
        X_S(i, :) = X0(i, DL + 1 - d_DL(i) : Nx - d_DL(i));
    end
    X = [X_S', Y'];
    
    phi=update_phi(Pi, m, v, W, beta, X, K, D);
    Nk=sum(phi, 1)';
    Pi=alpha0+Nk;
    beta=beta0+Nk;
    v=v0+Nk;
    m=update_m(phi, beta, m0, beta0, X, D);
    W=update_w(phi, beta, m, W0, beta0, m0, X, D);  

    %ELBO
    lb = elbo(phi, Pi, beta, v, W, alpha0, beta0, v0, W0);
    L = [L; lb];
    
end

t2 = cputime - t1;

figure
bar(d_DL)

figure(2)
plot(L,'b.-');

for i = 1: K
    Spred(:, :, i) = W(:, :, i) / v(i, :);
end

Pi_k = Pi./sum(Pi)
S = Spred; %S of the predictive

for i = 1: K
   S_in(:, :, i) = S(1: D - 1, 1: D - 1, i);
   S_out(:, :, i) = S(D, D, i);
   S_oi(:, :, i) = S(D, 1: D - 1, i);
end

for i = 1: K
   miu_in(i, :) = m(i, 1: D - 1);
   miu_out(i, :) = m(i, D);
end
