function fit_p=fitness(d_DL,X0,Y0,DL,w,tau)
N = size(Y0, 2);
D = size(w, 2);
for m = 1 : D
    X_s(m, :) = X0(m, DL + 1 - d_DL(m) : end - d_DL(m));
end
Q1 = N * (-1 * log(sqrt(2 * pi * tau)));
Q20 = sum((Y0 - w * X_s) .* (Y0 - w * X_s));
Q2 = (-N / (2 * tau)) * Q20;
fit_p = -(Q1 + Q2);
return;