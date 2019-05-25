function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================

theta = pinv(X'*X)*X'*y;

% ============================================================

end