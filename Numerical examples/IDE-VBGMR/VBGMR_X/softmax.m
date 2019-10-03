function sm = softmax(x)
ex = exp(x);
sm = ex ./ repmat(sum(ex, 2), 1, size(ex, 2));