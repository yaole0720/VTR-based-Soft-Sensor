%% 子程序1（就一句话，好像有点多此一举）：
function z = funGaussSample( mu, sigma, dim)
    %GAUSSAMPLE Summary of this function goes here
    %   Detailed explanation goes here

    %R = chol(sigma);
    %z = repmat(mu,dim(1),1) + randn(dim)*R;
    z = mvnrnd(mu, sigma, dim(1));

end