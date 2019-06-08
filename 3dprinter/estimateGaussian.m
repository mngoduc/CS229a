function [mu sigma2] = estimateGaussian(X)

[m, n] = size(X);           % Useful variables
mu = zeros(n, 1);           % Return these values correctly
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================

mu = mean(X,1);
sigma2 = (std(X,1)).^2;

% ============================================================
end