clc
clear all
close all

%% 数值例子
dim=[4, 216];   % k*N，k为数据维度，N为数据数目
Nclst=3;         % 簇的数量，也就是隐变量z可能取值的个数
k=dim(1);
N=dim(2);

alpha = [0.2, 0.3, 0.5];                   % z1点分布概率

mureal = [1  3  5;
         -3 -2 -1;
          3  2  5;
         -4 -6 -8];                 % p(x|z)的均值
sigreal=zeros(k,k,Nclst);                   % p(x|z)的方差
sigreal(:,:,1)=[0.4  -0.1  -0.1   0.2; 
                -0.1  0.3   0.1   0.05;
                -0.1  0.1   0.5   -0.1;
                0.2   0.05   -0.1   0.2];
sigreal(:,:,2)=[0.5   -0.1  -0.1  0.1; 
                -0.1   0.3  0.2   0.1;
                -0.1   0.2  0.3   -0.1;
                0.1    0.1  -0.1  0.4];
sigreal(:,:,3)=[0.5  -0.1   0.1   0.2;
                -0.1  0.6   0.05  0.1;
                 0.1  0.05  0.2   -0.1;
                 0.2  0.1   -0.1   0.4];

zreal = zeros(1,dim(2));                      % 隐变量z
x = zeros(dim);                               % 可观测到的变量x
for ii=1 : dim(2)
    zreal(ii) = randsrc(1,1,[1:Nclst; alpha]);   
    x(:,ii) = mvnrnd(mureal(:, zreal(ii)), squeeze(sigreal(:,:,zreal(ii))),1);
end

for i = 1 : Nclst
    label{i} = find(zreal==i);
end
X1 = [x(:, label{1}),x(:, label{2}),x(:, label{3})]';

X2 = X1;