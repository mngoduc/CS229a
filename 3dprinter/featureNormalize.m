function [X_norm, mu, sigma] = featureNormalize(X)

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================

mu = mean(X);
sigma = std(X);
numerator = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, numerator, sigma);
% ============================================================
end