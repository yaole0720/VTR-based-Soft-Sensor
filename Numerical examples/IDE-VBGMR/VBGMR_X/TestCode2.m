clc
clear all
close all

load X2;
load Work_TR_noVTR;

Data_X20 = X2(:, 1:3);
Data_Y20 = X2(:, 4);

N2 = size(Data_Y20, 1);
Data_X21 = [dx11; Data_X20(:, 1); dx12];
Data_X22 = [dx21;Data_X20(:, 2); dx22];
Data_X23 = [Data_X20(:, 3); dx3];
Data_Y21 = [dy; Data_Y20];

Data2 = [Data_X21,Data_X22,Data_X23,Data_Y21];

Data_X2 = Data2(d0(3)+1:N2, 1:3);
Data_Y2 = Data2(d0(3)+1:N2, 4);

%% ≤‚ ‘ ˝æ›
% Nb2 = 210;
% X_Ts = Data_X2(1 : Nb2, :)';
% Y_Ts = Data_Y2(1 : Nb2, :)';

X_Ts = X_tr;
Y_Ts = Y_tr;

xt = [X_Ts', Y_Ts'];
[Nt, Dt] = size(xt);

xt_in = xt(:, 1: Dt-1);
xt_out = xt(:, Dt);

for t = 1: Nt
    for i = 1: K
        miu_t_out(t, i) = miu_out(i, :) + S_oi(:, :, i) * inv(S_in(:, :, i)) * (xt_in(t, :) - miu_in(i, :))';
        hx(t, i) = Pi_k(i, :) * (1 / sqrt(det(S_in(:, :, i)))) * exp((-1 / 2) * (xt_in(t, :) - miu_in(i, :)) * inv(S_in(:, :, i)) * (xt_in(t, :) - miu_in(i, :))');
    end
    Hx(t, :) = hx(t, :) / sum(hx(t, :), 2);
    xt_out_pre(t,:) = Hx(t, :) * miu_t_out(t, :)';
end

RMSE = sqrt((xt_out - xt_out_pre)' * (xt_out - xt_out_pre) / Nt)
R2 = 1 - (sum((xt_out - xt_out_pre) .* (xt_out - xt_out_pre))) / (sum((xt_out - mean(xt_out)) .* (xt_out - mean(xt_out))))

figure
plot(xt_out, 'b.-');
hold on
plot(xt_out_pre, 'r.-');
