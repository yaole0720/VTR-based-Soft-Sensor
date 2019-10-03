clc 
clear all
close all

%%%% 生成数据
D = 5;N0 = 1020;xsigma = 1;esigma=1;st=1;DL=10;
d0=[3,4,5,6,7];
for m = 1:D
    d(m,:)=ones(1, N0-DL) * d0(m);
end
randn('state',st);
W = [3,4,5,9,12]; % 回归系数
X00 = randn(D, N0) * xsigma; % 采集的数据序列
eps = randn(1, N0) * esigma;
for i=1:N0-DL
    for m = 1: D
        X(m,i)=X00(m,DL+i-d(m, i)); % 正常数据序列
    end
    Y(i)=W*X(:,i)+eps(i); % 由正常数据序列产生的Y
end
Data_X = X00(:, DL+1:end);
Data_Y = Y;

%% 有延时补偿最小二乘 %%%
Nb = 800;
X_Tr = Data_X(:, 1:Nb);
Y_Tr = Data_Y(:, 1:Nb);
X0 = X_Tr;
Y0 = Y_Tr(:, DL + 1 :end);
[D, N] = size(X0);

%%%%% DE寻优
NP=100;
gen_max=50;
w=rand(1,D);tau=rand(1,1);
Iter = 30;
for i = 1 : Iter
    i
    [Pb,d_DL,x,trace]=IDE(D,DL,NP,gen_max,X0,Y0,w,tau);
    for m = 1 : D
        X_S(m, :) = X0(m, DL + 1 - d_DL(m) : N - d_DL(m));
    end
    w = (Y0 * X_S') * inv(X_S * X_S');
    tau = (1/(N - DL)) * sum((Y0 - w * X_S) .* (Y0 - w * X_S));
end

figure
bar(d_DL)

MLE = -1 * (trace(:, 2));
figure
plot(MLE, '.-')

X_Ts = Data_X(:, Nb+1:end);
Y_Ts = Data_Y(:, Nb+1:end);

% X_Ts = X_Tr;
% Y_Ts = Y_Tr;
Nts = size(X_Ts, 2);
for m = 1 : D
    X_TS(m, :) = X_Ts(m, DL + 1 - d_DL(m) : Nts - d_DL(m));
end
Y_TS = Y_Ts(:, DL + 1 : end);
Yp_Ts = w * X_TS;
RMSE_p = sqrt(sum((Yp_Ts - Y_TS) .* (Yp_Ts - Y_TS))/(Nts-DL)); %%测试RMSE
R2_p = 1 - (sum((Yp_Ts - Y_TS) .* (Yp_Ts - Y_TS))) / (sum((Y_TS - mean(Y_TS)) .* (Y_TS - mean(Y_TS))));
figure
plot(Y_TS, 'bo-');
hold on
plot(Yp_Ts, 'r.-');

%% 无时延补偿最小二乘法 %%%
X0_v = X_Tr(:, 1:end);
Y_v = Y_Tr(:, DL + 1 :end);
[D, N_v] = size(X0_v);

for m = 1 : D
    X_S_v(m, :) = X0_v(m, DL + 1 : N_v);
end
w_v = (Y_v * X_S_v') * inv(X_S_v * X_S_v');
tau_v = (1/(N_v - DL)) * sum((Y_v - w_v * X_S_v) .* (Y_v - w_v * X_S_v));

% 测试

Nts_v = size(X_Ts, 2);
for m = 1 : D
    X_TS_v(m, :) = X_Ts(m, DL + 1: Nts_v);
end
Y_TS_v = Y_Ts(:, DL + 1 : end);
Yp_Ts_v = w_v * X_TS_v;
RMSE_p_v = sqrt(sum((Yp_Ts_v - Y_TS_v) .* (Yp_Ts_v - Y_TS_v))/(Nts_v-DL)); %%测试RMSE
R2_p_v = 1 - (sum((Yp_Ts_v - Y_TS_v) .* (Yp_Ts_v - Y_TS_v))) / (sum((Y_TS_v - mean(Y_TS_v)) .* (Y_TS_v - mean(Y_TS_v))));

figure
plot(Y_TS_v, 'bo-');
hold on
plot(Yp_Ts_v, 'r.-');

