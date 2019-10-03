function MG = multigammaln(a, p)
MG1 = (1 / 4) * p * (p-1) * log(pi);
for j = 1: p
    MG2(j)= gammaln(a + (1 - j) / 2);
end
MG = MG1 + sum(MG2);
