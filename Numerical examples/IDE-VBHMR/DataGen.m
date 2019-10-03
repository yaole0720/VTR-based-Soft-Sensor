clc
clear all
close all

%% ��ֵ����
dim=[4,3000];   % k*N��kΪ����ά�ȣ�NΪ������Ŀ
Nclst=3;         % �ص�������Ҳ����������z����ȡֵ�ĸ���
k=dim(1);
N=dim(2);
Aerr=1e-4;       % ���ڵ�����ֹ

A0real = [0.3, 0.3, 0.4];                   % z1��ֲ�����
Areal = [0.8, 0.1, 0.1;...
         0.2, 0.7, 0.1;...
         0.0, 0.5, 0.5];                    % z����ת�Ƹ��ʾ���

mureal = [1  3  5;
         -3 -2 -1;
          3  2  5;
         -4 -6 -8];                 % p(x|z)�ľ�ֵ
sigreal=zeros(k,k,Nclst);                   % p(x|z)�ķ���
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

zreal = zeros(1,dim(2));                      % ������z
zreal(1) = randsrc(1,1,[1:Nclst;A0real]);
x = zeros(dim);                               % �ɹ۲⵽�ı���x
x(:,1) = funGaussSample(mureal(:,zreal(1)),squeeze(sigreal(:,:,zreal(1))),1);
for ii=2 : dim(2)
    zreal(ii) = randsrc(1,1,[1:Nclst; Areal(zreal(ii - 1),:)]);   
    x(:,ii) = funGaussSample(mureal(:, zreal(ii)), squeeze(sigreal(:,:,zreal(ii))),1);
end
X = x';
figure
plot(X(:,1),X(:,4),'b*')